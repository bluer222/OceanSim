# Omniverse Import
import omni.replicator.core as rep
from omni.replicator.core.scripts.functional import write_image
import omni.graph.core as og
import omni.usd
import omni.timeline
import omni.ui as ui

# Isaac sim import
from isaacsim.sensors.camera import Camera
import numpy as np
import warp as wp
import yaml
import carb

# Custom import
from isaacsim.mysim.utils.UWrenderer_utils import UW_render, rgba_to_greyscale


class UW_Camera(Camera):

    def __init__(self, 
                 prim_path, 
                 name = "UW_Camera", 
                 frequency = None, 
                 dt = None, 
                 resolution = None, 
                 position = None, 
                 orientation = None, 
                 translation = None, 
                 render_product_path = None,
                 ros2_publish = False,
                 greyscale = False):
        
        """Initialize an underwater camera sensor.
    
        Args:
            prim_path (str): prim path of the Camera Prim to encapsulate or create.
            name (str, optional): shortname to be used as a key by Scene class.
                                    Note: needs to be unique if the object is added to the Scene.
                                    Defaults to "UW_Camera".
            frequency (Optional[int], optional): Frequency of the sensor (i.e: how often is the data frame updated).
                                                Defaults to None.
            dt (Optional[str], optional): dt of the sensor (i.e: period at which a the data frame updated). Defaults to None.
            resolution (Optional[Tuple[int, int]], optional): resolution of the camera (width, height). Defaults to None.
            position (Optional[Sequence[float]], optional): position in the world frame of the prim. shape is (3, ).
                                                        Defaults to None, which means left unchanged.
            translation (Optional[Sequence[float]], optional): translation in the local frame of the prim
                                                            (with respect to its parent prim). shape is (3, ).
                                                            Defaults to None, which means left unchanged.
            orientation (Optional[Sequence[float]], optional): quaternion orientation in the world/ local frame of the prim
                                                            (depends if translation or position is specified).
                                                            quaternion is scalar-first (w, x, y, z). shape is (4, ).
                                                            Defaults to None, which means left unchanged.
            render_product_path (str): path to an existing render product, will be used instead of creating a new render product
                                    the resolution and camera attached to this render product will be set based on the input arguments.
                                    Note: Using same render product path on two Camera objects with different camera prims, resolutions is not supported
                                    Defaults to None
        """
        self._name = name
        self._prim_path = prim_path
        self._res = resolution
        self._writing = False
        self._ros2_publish = ros2_publish
        self._ros2_graph_path = None
        self._ros2_node = None
        self._ros2_image_pub = None
        self._image_msg_type = None
        self._greyscale = greyscale

        super().__init__(prim_path, name, frequency, dt, resolution, position, orientation, translation, render_product_path)

    def initialize(self, 
                   UW_param: np.ndarray = np.array([0.0, 0.31, 0.24, 0.05, 0.05, 0.2, 0.05, 0.05, 0.05 ]),
                   viewport: bool = True,
                   writing_dir: str = None,
                   UW_yaml_path: str = None,
                   physics_sim_view=None):
        
        """Configure underwater rendering properties and initialize pipelines.
    
        Args:
            UW_param (np.ndarray, optional): Underwater parameters array:
                [0:3] - Backscatter value (RGB)
                [3:6] - Attenuation coefficients (RGB)
                [6:9] - Backscatter coefficients (RGB)
                Defaults to typical coastal water values.
            viewport (bool, optional): Enable viewport visualization. Defaults to True.
            writing_dir (str, optional): Directory to save rendered images. Defaults to None.
            UW_yaml_path (str, optional): Path to YAML file with water properties. Defaults to None.
            physics_sim_view (_type_, optional): _description_. Defaults to None.            
    
        """
        self._id = 0
        self._viewport = viewport
        self._device = wp.get_preferred_device()
        super().initialize(physics_sim_view)

        if UW_yaml_path is not None:
            with open(UW_yaml_path, 'r') as file:
                try:
                    # Load the YAML content
                    yaml_content = yaml.safe_load(file)
                    self._backscatter_value = wp.vec3f(*yaml_content['backscatter_value'])
                    self._atten_coeff = wp.vec3f(*yaml_content['atten_coeff'])
                    self._backscatter_coeff = wp.vec3f(*yaml_content['backscatter_coeff'])
                    print(f"[{self._name}] On {str(self._device)}. Using loaded render parameters:")
                    print(f"[{self._name}] Render parameters: {yaml_content}")
                except yaml.YAMLError as exc:
                    carb.log_error(f"[{self._name}] Error reading YAML file: {exc}")
        else:
            self._backscatter_value = wp.vec3f(*UW_param[0:3])
            self._atten_coeff = wp.vec3f(*UW_param[6:9])
            self._backscatter_coeff = wp.vec3f(*UW_param[3:6])
            print(f'[{self._name}] On {str(self._device)}. Using default render parameters.')

        
        self._rgba_annot = rep.AnnotatorRegistry.get_annotator('LdrColor', device=str(self._device))
        self._depth_annot = rep.AnnotatorRegistry.get_annotator('distance_to_camera', device=str(self._device))

        self._rgba_annot.attach(self._render_product_path)
        self._depth_annot.attach(self._render_product_path)

        if self._viewport:
            self.make_viewport()

        if writing_dir is not None:
            self._writing = True
            self._writing_backend = rep.BackendDispatch({"paths": {"out_dir": writing_dir}})

        if self._ros2_publish:
            self._setup_ros2_graph()
            self._setup_ros2_image_publisher()
        
        print(f'[{self._name}] Initialized successfully. Data writing: {self._writing}. ROS2: {self._ros2_publish}')
    
    def render(self):
        """Process and display a single frame with underwater effects.
    
        Note:
            - Updates viewport display if enabled
            - Saves image to disk if writing_dir was specified
        """
        raw_rgba = self._rgba_annot.get_data()
        depth = self._depth_annot.get_data()
        if raw_rgba.size !=0:
            uw_image = wp.zeros_like(raw_rgba)
            wp.launch(
                dim=np.flip(self.get_resolution()),
                kernel=UW_render,
                inputs=[
                    raw_rgba,
                    depth,
                    self._backscatter_value,
                    self._atten_coeff,
                    self._backscatter_coeff
                ],
                outputs=[
                    uw_image
                ]
            )

            if self._greyscale:
                wp.launch(
                    dim=np.flip(self.get_resolution()),
                    kernel=rgba_to_greyscale,
                    inputs=[uw_image]
                )

            if self._ros2_publish:
                self._publish_ros2_image(uw_image)

            if self._viewport:
                self._provider.set_bytes_data_from_gpu(uw_image.ptr, self.get_resolution())
            if self._writing:
                self._writing_backend.schedule(write_image, path=f'UW_image_{self._id}.png', data=uw_image)
                print(f'[{self._name}] [{self._id}] Rendered image saved to {self._writing_backend.output_dir}')

            self._id += 1

    def make_viewport(self):
        """Create a viewport window for real-time visualization.
    
        Note:
            - Window size fixed at 1280x760 pixels
        """
    
        self.wrapped_ui_elements = []
        self.window = ui.Window(self._name, width=1280, height=720 + 40, visible=True)
        self._provider = ui.ByteImageProvider()
        with self.window.frame:
            with ui.ZStack(height=720):
                ui.Rectangle(style={"background_color": 0xFF000000})
                ui.Label('Run the scenario for image to be received',
                         style={'font_size': 55,'alignment': ui.Alignment.CENTER},
                         word_wrap=True)
                image_provider = ui.ImageWithProvider(self._provider, width=1280, height=720,
                                     style={'fill_policy': ui.FillPolicy.PRESERVE_ASPECT_FIT,
                                    'alignment' :ui.Alignment.CENTER})
        
        self.wrapped_ui_elements.append(image_provider)
        self.wrapped_ui_elements.append(self._provider)
        self.wrapped_ui_elements.append(self.window)

    # Detach the annotator from render product and clear the data cache
    def close(self):
        """Clean up resources by detaching annotators and clearing caches.
    
        Note:
            - Required for proper shutdown when done using the sensor
            - Also closes viewport window if one was created
            - Removes the ROS2 OmniGraph if ros2_publish was enabled
        """
        self._rgba_annot.detach(self._render_product_path)
        self._depth_annot.detach(self._render_product_path)

        rep.AnnotatorCache.clear(self._rgba_annot)
        rep.AnnotatorCache.clear(self._depth_annot)

        if self._viewport:
            self.ui_destroy()

        if self._ros2_node is not None:
            self._ros2_node.destroy_node()
            self._ros2_node = None
            self._ros2_image_pub = None
            self._image_msg_type = None

        if self._ros2_graph_path is not None:
            stage = omni.usd.get_context().get_stage()
            if stage and stage.GetPrimAtPath(self._ros2_graph_path).IsValid():
                stage.RemovePrim(self._ros2_graph_path)
                print(f'[{self._name}] ROS2 graph removed: {self._ros2_graph_path}')
            self._ros2_graph_path = None
            
        print(f'[{self._name}] Annotator detached. AnnotatorCache cleaned.')
    
    
    def ui_destroy(self):
        """Explicitly destroy viewport UI elements.
    
        Note:
            - Called automatically by close()
            - Only needed if manually managing UI lifecycle
        """
        for elem in self.wrapped_ui_elements:
            elem.destroy()

    def _setup_ros2_image_publisher(self):
        """Create a Python ROS2 publisher for the post-processed underwater image."""
        try:
            import rclpy
            from sensor_msgs.msg import Image

            if not rclpy.ok():
                rclpy.init(args=None)

            safe_name = self._name.replace(' ', '_').replace('-', '_')
            self._ros2_node = rclpy.create_node(f"{safe_name}_uw_image_publisher")
            self._ros2_image_pub = self._ros2_node.create_publisher(Image, f"{self._name}/image_raw", 10)
            self._image_msg_type = Image
        except Exception as exc:
            self._ros2_node = None
            self._ros2_image_pub = None
            self._image_msg_type = None
            carb.log_error(f"[{self._name}] Failed to create ROS2 image publisher: {exc}")
            raise

    def _publish_ros2_image(self, uw_image):
        """Publish the processed underwater image as sensor_msgs/Image."""
        if self._ros2_image_pub is None or self._image_msg_type is None:
            return

        wp.synchronize()
        image = np.asarray(uw_image.numpy())
        height, width = image.shape[:2]

        msg = self._image_msg_type()
        sim_time = omni.timeline.get_timeline_interface().get_current_time()
        msg.header.stamp.sec = int(sim_time)
        msg.header.stamp.nanosec = int((sim_time - msg.header.stamp.sec) * 1_000_000_000)
        msg.header.frame_id = self._name
        msg.height = height
        msg.width = width
        msg.is_bigendian = 0

        if self._greyscale:
            mono = np.ascontiguousarray(image[:, :, 0])
            msg.encoding = "mono8"
            msg.step = width
            msg.data = mono.tobytes()
        else:
            rgb = np.ascontiguousarray(image[:, :, :3])
            msg.encoding = "rgb8"
            msg.step = width * 3
            msg.data = rgb.tobytes()

        self._ros2_image_pub.publish(msg)

    def _setup_ros2_graph(self):
        """Build an OmniGraph that publishes camera info to ROS2.

        The processed underwater image is published from render() because it is
        generated after the render-product pipeline.
        """
        keys = og.Controller.Keys
        safe_name = self._name.replace(' ', '_').replace('-', '_')
        self._ros2_graph_path = f"/ActionGraph_{safe_name}"

        try:
            graph, _, _, _ = og.Controller.edit(
                {"graph_path": self._ros2_graph_path, "evaluator_name": "execution"},
                {
                    keys.CREATE_NODES: [
                        ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
                        ("Context", "isaacsim.ros2.bridge.ROS2Context"),
                        ("CameraInfo", "isaacsim.ros2.bridge.ROS2CameraInfoHelper"),
                    ],
                    keys.SET_VALUES: [
                        ("CameraInfo.inputs:renderProductPath", self._render_product_path),
                        ("CameraInfo.inputs:topicName", f"{self._name}/camera_info"),
                        ("CameraInfo.inputs:frameId", self._name),
                        ("CameraInfo.inputs:resetSimulationTimeOnStop", True),
                    ],
                    keys.CONNECT: [
                        ("OnPlaybackTick.outputs:tick", "CameraInfo.inputs:execIn"),
                        ("Context.outputs:context", "CameraInfo.inputs:context"),
                    ],
                },
            )
            og.Controller.evaluate_sync(graph)
        except Exception as exc:
            graph_path = self._ros2_graph_path
            self._ros2_graph_path = None
            carb.log_error(f"[{self._name}] Failed to create ROS2 camera info graph at {graph_path}: {exc}")
            raise

        stage = omni.usd.get_context().get_stage()
        if not stage or not stage.GetPrimAtPath(self._ros2_graph_path).IsValid():
            graph_path = self._ros2_graph_path
            self._ros2_graph_path = None
            raise RuntimeError(f"[{self._name}] ROS2 graph edit completed but no prim exists at {graph_path}")

        print(f"[{self._name}] ROS2 camera info graph created at {self._ros2_graph_path}")
        print(f"[{self._name}] Publishing processed image: /{self._name}/image_raw  |  camera info: /{self._name}/camera_info")
