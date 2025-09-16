"""Interactive GUI application for URDF spherization using Viser."""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import viser
import yourdfpy
from robot_descriptions.loaders.yourdfpy import load_robot_description

from .core import (
    EnhancedViserUrdf,
    Geometry,
    GeometryStore,
    inject_geometries_into_urdf_xml,
    inject_spheres_into_urdf_xml,
)


class BubblifyApp:
    """Main application class for interactive URDF spherization."""

    def __init__(
        self,
        robot_name: str = "panda",
        urdf_path: Optional[Path] = None,
        show_collision: bool = False,
        port: int = 8080,
        spherization_yml: Optional[Path] = None,
    ):
        """Initialize the Bubblify application.

        Args:
            robot_name: Name of robot from robot_descriptions (used if urdf_path is None)
            urdf_path: Path to custom URDF file
            show_collision: Whether to show collision meshes
            port: Viser server port
            spherization_yml: Path to existing spherization YAML file to load
        """
        self.server = viser.ViserServer(port=port)
        self.show_collision = show_collision

        # Load URDF
        if urdf_path is not None:
            self.urdf = yourdfpy.URDF.load(
                str(urdf_path),  # urdf_path,
                build_scene_graph=True,
                load_meshes=True,
                build_collision_scene_graph=show_collision,
                load_collision_meshes=show_collision,
            )
            self.urdf_path = urdf_path
        else:
            self.urdf = load_robot_description(
                robot_name + "_description",
                load_meshes=True,
                build_scene_graph=True,
                load_collision_meshes=show_collision,
                build_collision_scene_graph=show_collision,
            )
            self.urdf_path = None

        # Enhanced URDF visualizer with per-link control
        self.urdf_viz = EnhancedViserUrdf(
            self.server,
            urdf_or_path=self.urdf,
            load_meshes=True,
            load_collision_meshes=show_collision,
            collision_mesh_color_override=(1.0, 0.0, 0.0, 0.4),
        )

        # Geometry management
        self.geometry_store = GeometryStore()
        # Keep backward compatibility alias
        self.sphere_store = self.geometry_store

        # GUI state
        self.current_geometry_id: Optional[int] = None
        # Keep backward compatibility alias
        self.current_sphere_id: Optional[int] = None
        self.current_link: str = ""
        self.joint_sliders: List[viser.GuiInputHandle[float]] = []
        self.transform_control: Optional[viser.TransformControlsHandle] = None
        self.radius_gizmo: Optional[viser.TransformControlsHandle] = None

        # Box resize gizmos (one for each axis: X, Y, Z)
        self.box_resize_gizmos: Dict[str, viser.TransformControlsHandle] = {}

        # GUI control references for syncing
        self._link_dropdown = None
        self._current_link_dropdown = None
        self._geometry_dropdown = None
        self._geometry_type_dropdown = None
        self._sphere_radius_slider = None
        self._box_size_sliders = None
        self._cylinder_radius_slider = None
        self._cylinder_height_slider = None
        self._geometry_color_input = None

        # Keep backward compatibility aliases
        self._sphere_dropdown = None
        self._sphere_color_input = None

        # Flag to prevent recursive updates
        self._updating_geometry_ui = False
        # Keep backward compatibility alias
        self._updating_sphere_ui = False

        # Visibility settings
        self.show_selected_link: bool = True
        self.show_other_links: bool = True

        # Sphere opacity settings
        self.selected_sphere_opacity: float = 1.0
        self.unselected_spheres_opacity: float = 0.5
        self.other_links_spheres_opacity: float = 0.2

        # Create sphere root frame
        self.spheres_root = self.server.scene.add_frame("/spheres", show_axes=False)

        # Setup GUI
        self._setup_robot_controls()
        self._setup_visibility_controls()
        self._setup_sphere_controls()
        self._setup_export_controls()

        # Add a grid for reference
        self._add_reference_grid()

        # Initialize visibility states
        self._update_mesh_visibility()

        # Load spherization YAML if provided
        if spherization_yml is not None:
            self._load_spherization_yaml(spherization_yml)

        print(f"🎯 Bubblify server running at http://localhost:{port}")
        print("Use the GUI controls to add and edit collision spheres!")

    def _setup_robot_controls(self):
        """Setup robot configuration and visibility controls."""
        with self.server.gui.add_folder("🤖 Robot Controls"):
            # Joint sliders
            initial_config = []

            for joint_name, (
                lower,
                upper,
            ) in self.urdf_viz.get_actuated_joint_limits().items():
                lower = lower if lower is not None else -np.pi
                upper = upper if upper is not None else np.pi
                initial_pos = (
                    0.0 if lower < -0.1 and upper > 0.1 else (lower + upper) / 2.0
                )

                slider = self.server.gui.add_slider(
                    label=joint_name,
                    min=lower,
                    max=upper,
                    step=1e-3,
                    initial_value=initial_pos,
                )
                self.joint_sliders.append(slider)
                initial_config.append(initial_pos)

            # Connect sliders to URDF update
            def update_robot_config():
                config = np.array([s.value for s in self.joint_sliders])
                self.urdf_viz.update_cfg(config)

            for slider in self.joint_sliders:
                slider.on_update(lambda _: update_robot_config())

            # Apply initial configuration
            update_robot_config()

            # Reset button
            reset_joints_btn = self.server.gui.add_button("🏠 Reset to Home")

            @reset_joints_btn.on_click
            def _(_):
                for slider, init_val in zip(self.joint_sliders, initial_config):
                    slider.value = init_val

    def _setup_visibility_controls(self):
        """Setup visibility controls in separate section."""
        with self.server.gui.add_folder("👁️ Visibility Controls"):
            # Current link dropdown
            all_links = self.urdf_viz.get_all_link_names()
            if not all_links:
                all_links = ["base_link"]
            current_link_dropdown = self.server.gui.add_dropdown(
                "Current Link", options=all_links, initial_value=all_links[0]
            )

            # Mesh visibility toggles
            show_selected_link_cb = self.server.gui.add_checkbox(
                "Show Selected Link", initial_value=self.show_selected_link
            )
            show_other_links_cb = self.server.gui.add_checkbox(
                "Show Other Links", initial_value=self.show_other_links
            )

            # Sphere opacity controls with clearer names
            selected_sphere_opacity = self.server.gui.add_slider(
                "Current Sphere",
                min=0.0,
                max=1.0,
                step=0.1,
                initial_value=self.selected_sphere_opacity,
            )
            unselected_spheres_opacity = self.server.gui.add_slider(
                "Other Spheres (Same Link)",
                min=0.0,
                max=1.0,
                step=0.1,
                initial_value=self.unselected_spheres_opacity,
            )
            other_links_spheres_opacity = self.server.gui.add_slider(
                "Spheres (Other Links)",
                min=0.0,
                max=1.0,
                step=0.1,
                initial_value=self.other_links_spheres_opacity,
            )

            # Store references for updates
            self._current_link_dropdown = current_link_dropdown

            # Set initial current link from dropdown
            self.current_link = current_link_dropdown.value

            @current_link_dropdown.on_update
            def _(_):
                self.current_link = current_link_dropdown.value
                self._sync_link_selection()
                self._update_mesh_visibility()

            @show_selected_link_cb.on_update
            def _(_):
                self.show_selected_link = show_selected_link_cb.value
                self._update_mesh_visibility()

            @show_other_links_cb.on_update
            def _(_):
                self.show_other_links = show_other_links_cb.value
                self._update_mesh_visibility()

            @selected_sphere_opacity.on_update
            def _(_):
                self.selected_sphere_opacity = selected_sphere_opacity.value
                self._update_sphere_opacities()

            @unselected_spheres_opacity.on_update
            def _(_):
                self.unselected_spheres_opacity = unselected_spheres_opacity.value
                self._update_sphere_opacities()

            @other_links_spheres_opacity.on_update
            def _(_):
                self.other_links_spheres_opacity = other_links_spheres_opacity.value
                self._update_sphere_opacities()

    def _setup_sphere_controls(self):
        """Setup geometry creation and editing controls."""
        with self.server.gui.add_folder("🔶 Geometry Editor"):
            # Get links for dropdown
            all_links = self.urdf_viz.get_all_link_names()
            if not all_links:
                all_links = ["base_link"]

            # Link selection
            link_dropdown = self.server.gui.add_dropdown(
                "Link", options=all_links, initial_value=all_links[0]
            )
            self.current_link = link_dropdown.value
            self._link_dropdown = link_dropdown  # Store reference for syncing

            # Geometry type selection
            geometry_type_dropdown = self.server.gui.add_dropdown(
                "Geometry Type",
                options=["sphere", "box", "cylinder"],
                initial_value="sphere",
            )
            self._geometry_type_dropdown = geometry_type_dropdown

            # Geometry selection dropdown (will be populated based on selected link)
            geometry_dropdown = self.server.gui.add_dropdown(
                "Geometry", options=["None"], initial_value="None"
            )
            self._geometry_dropdown = geometry_dropdown  # Store reference
            # Keep backward compatibility alias
            self._sphere_dropdown = geometry_dropdown

            # Geometry creation and deletion
            add_geometry_btn = self.server.gui.add_button("➕ Add Geometry")
            delete_geometry_btn = self.server.gui.add_button("🗑️ Delete Selected")

            # Geometry statistics
            total_geometry_count = self.server.gui.add_text(
                "Total Geometries", initial_value="0"
            )
            link_geometry_count = self.server.gui.add_text(
                "Geometries on Current Link", initial_value="0"
            )

            # Geometry properties - Sphere
            with self.server.gui.add_folder(
                "⚪ Sphere Properties", expand_by_default=True
            ):
                sphere_radius = self.server.gui.add_slider(
                    "Radius", min=0.005, max=0.14, step=0.001, initial_value=0.05
                )
                self._sphere_radius_slider = sphere_radius

            # Geometry properties - Box
            with self.server.gui.add_folder(
                "📦 Box Properties", expand_by_default=False
            ):
                box_length = self.server.gui.add_slider(
                    "Length", min=0.01, max=0.5, step=0.001, initial_value=0.1
                )
                box_width = self.server.gui.add_slider(
                    "Width", min=0.01, max=0.5, step=0.001, initial_value=0.1
                )
                box_height = self.server.gui.add_slider(
                    "Height", min=0.01, max=0.5, step=0.001, initial_value=0.1
                )
                self._box_size_sliders = (box_length, box_width, box_height)

            # Geometry properties - Cylinder
            with self.server.gui.add_folder(
                "🥫 Cylinder Properties", expand_by_default=False
            ):
                cylinder_radius = self.server.gui.add_slider(
                    "Radius", min=0.005, max=0.14, step=0.001, initial_value=0.05
                )
                cylinder_height = self.server.gui.add_slider(
                    "Height", min=0.01, max=0.5, step=0.001, initial_value=0.1
                )
                self._cylinder_radius_slider = cylinder_radius
                self._cylinder_height_slider = cylinder_height

            # Common properties
            geometry_color = self.server.gui.add_rgb(
                "Color", initial_value=(255, 180, 60)
            )
            self._geometry_color_input = geometry_color
            # Keep backward compatibility alias
            self._sphere_color_input = geometry_color

            # Rotation properties
            with self.server.gui.add_folder(
                "🔄 Rotation Properties", expand_by_default=False
            ):
                roll_slider = self.server.gui.add_slider(
                    "Roll (X)", min=-3.14159, max=3.14159, step=0.01, initial_value=0.0
                )
                pitch_slider = self.server.gui.add_slider(
                    "Pitch (Y)", min=-3.14159, max=3.14159, step=0.01, initial_value=0.0
                )
                yaw_slider = self.server.gui.add_slider(
                    "Yaw (Z)", min=-3.14159, max=3.14159, step=0.01, initial_value=0.0
                )
                self._rpy_sliders = (roll_slider, pitch_slider, yaw_slider)

            def update_geometry_dropdown():
                """Update geometry dropdown based on selected link."""
                link_name = link_dropdown.value
                self.current_link = link_name
                geometries = self.geometry_store.get_geometries_for_link(link_name)

                if geometries:
                    options = [f"{g.geometry_type.title()} {g.id}" for g in geometries]
                    geometry_dropdown.options = options

                    # Determine which geometry to select
                    geometry_to_select = None
                    if self.current_geometry_id is not None:
                        # Try to keep current selection if it's still valid for this link
                        current_geometry = self.geometry_store.by_id.get(
                            self.current_geometry_id
                        )
                        if current_geometry and current_geometry.link == link_name:
                            geometry_to_select = current_geometry

                    # If no valid current selection, select the first geometry
                    if geometry_to_select is None:
                        geometry_to_select = geometries[0]

                    # Update both dropdown and current_geometry_id
                    geometry_dropdown.value = f"{geometry_to_select.geometry_type.title()} {geometry_to_select.id}"
                    self.current_geometry_id = geometry_to_select.id
                    # Keep backward compatibility
                    self.current_sphere_id = geometry_to_select.id
                else:
                    geometry_dropdown.options = ["None"]
                    geometry_dropdown.value = "None"
                    self.current_geometry_id = None
                    self.current_sphere_id = None

                self._update_transform_control()
                self._update_radius_gizmo()
                self._update_box_resize_gizmos()
                self._update_geometry_properties_ui()
                self._update_sphere_opacities()
                self._update_mesh_visibility()

            # Keep backward compatibility alias
            update_sphere_dropdown = update_geometry_dropdown

            def update_selected_geometry():
                """Update selected geometry ID from dropdown and switch link context."""
                if geometry_dropdown.value != "None":
                    geometry_id = int(geometry_dropdown.value.split()[-1])
                    self.current_geometry_id = geometry_id
                    self.current_sphere_id = geometry_id  # Backward compatibility

                    # Get the geometry and switch to its link
                    if geometry_id in self.geometry_store.by_id:
                        geometry = self.geometry_store.by_id[geometry_id]
                        # Update the link context and sync dropdowns
                        if geometry.link != self.current_link:
                            self.current_link = geometry.link
                            link_dropdown.value = geometry.link
                            self._sync_link_selection()
                            self._update_mesh_visibility()

                    self._update_transform_control()
                    self._update_radius_gizmo()
                    self._update_box_resize_gizmos()
                    self._update_geometry_properties_ui()
                    self._update_sphere_opacities()
                else:
                    self.current_geometry_id = None
                    self.current_sphere_id = None
                    self._remove_transform_control()

            # Keep backward compatibility alias
            update_selected_sphere = update_selected_geometry

            @link_dropdown.on_update
            def _(_):
                self.current_link = link_dropdown.value
                self._sync_link_selection()
                self._update_mesh_visibility()
                update_geometry_dropdown()

            @geometry_dropdown.on_update
            def _(_):
                update_selected_geometry()

            @add_geometry_btn.on_click
            def _(_):
                """Add a new geometry to the selected link using current parameters."""
                link_name = link_dropdown.value
                geometry_type = geometry_type_dropdown.value

                # Get current parameters based on geometry type
                if geometry_type == "sphere":
                    geometry = self.geometry_store.add(
                        link_name,
                        xyz=(0.0, 0.0, 0.0),
                        geometry_type="sphere",
                        radius=sphere_radius.value,
                    )
                elif geometry_type == "box":
                    geometry = self.geometry_store.add(
                        link_name,
                        xyz=(0.0, 0.0, 0.0),
                        geometry_type="box",
                        size=(box_length.value, box_width.value, box_height.value),
                    )
                elif geometry_type == "cylinder":
                    geometry = self.geometry_store.add(
                        link_name,
                        xyz=(0.0, 0.0, 0.0),
                        geometry_type="cylinder",
                        cylinder_radius=cylinder_radius.value,
                        cylinder_height=cylinder_height.value,
                    )

                self._create_geometry_visualization(geometry)

                # Select the new geometry as current
                self.current_geometry_id = geometry.id
                self.current_sphere_id = geometry.id  # Backward compatibility

                # Update dropdown and controls immediately
                update_geometry_dropdown()

                # Directly call the control update methods to show gizmo immediately
                self._update_transform_control()
                self._update_radius_gizmo()
                self._update_box_resize_gizmos()
                self._update_geometry_properties_ui()
                self._update_sphere_opacities()

            @delete_geometry_btn.on_click
            def _(_):
                """Delete the selected geometry."""
                if self.current_geometry_id is not None:
                    self.geometry_store.remove(self.current_geometry_id)
                    self.current_geometry_id = None
                    self.current_sphere_id = None
                    self._remove_transform_control()
                    update_geometry_dropdown()

            def update_geometry_properties():
                """Update geometry properties from UI."""
                # Skip update if we're currently updating the UI to prevent recursive changes
                if self._updating_geometry_ui or self._updating_sphere_ui:
                    return

                if (
                    self.current_geometry_id is not None
                    and self.current_geometry_id in self.geometry_store.by_id
                ):
                    geometry = self.geometry_store.by_id[self.current_geometry_id]

                    # Update properties based on geometry type
                    if geometry.geometry_type == "sphere":
                        geometry.radius = float(sphere_radius.value)
                    elif geometry.geometry_type == "box":
                        geometry.size = (
                            box_length.value,
                            box_width.value,
                            box_height.value,
                        )
                    elif geometry.geometry_type == "cylinder":
                        geometry.cylinder_radius = float(cylinder_radius.value)
                        geometry.cylinder_height = float(cylinder_height.value)

                    # Update rotation properties only for non-sphere geometries
                    if geometry.geometry_type != "sphere":
                        new_rpy = (
                            self._rpy_sliders[0].value,  # roll
                            self._rpy_sliders[1].value,  # pitch
                            self._rpy_sliders[2].value,  # yaw
                        )
                        geometry.update_quaternion_from_rpy(new_rpy)

                        # Update transform control if active
                        if self.transform_control is not None:
                            self.transform_control.wxyz = geometry.local_wxyz

                    # Update common properties
                    geometry.color = tuple(int(c) for c in geometry_color.value)
                    self._update_geometry_visualization(geometry)
                    self._update_radius_gizmo()
                    self._update_box_resize_gizmos()

            # Keep backward compatibility alias
            update_sphere_properties = update_geometry_properties

            # Connect all property sliders to update function
            sphere_radius.on_update(lambda _: update_geometry_properties())
            box_length.on_update(lambda _: update_geometry_properties())
            box_width.on_update(lambda _: update_geometry_properties())
            box_height.on_update(lambda _: update_geometry_properties())
            cylinder_radius.on_update(lambda _: update_geometry_properties())
            cylinder_height.on_update(lambda _: update_geometry_properties())
            geometry_color.on_update(lambda _: update_geometry_properties())
            # Connect rotation sliders (will only affect non-sphere geometries)
            roll_slider.on_update(lambda _: update_geometry_properties())
            pitch_slider.on_update(lambda _: update_geometry_properties())
            yaw_slider.on_update(lambda _: update_geometry_properties())

            # Initialize
            update_geometry_dropdown()

            # Set up initial opacity state
            self._update_sphere_opacities()

    def _setup_export_controls(self):
        """Setup export functionality."""
        with self.server.gui.add_folder("💾 Export"):
            # Get default export names based on URDF
            default_name = "spherized"
            if self.urdf_path and self.urdf_path.stem:
                default_name = f"{self.urdf_path.stem}_spherized"

            # Export name configuration (no paths, just filenames)
            export_name_input = self.server.gui.add_text(
                "Export Name", initial_value=default_name
            )

            # Export options
            export_yml_btn = self.server.gui.add_button("Export Spheres (YAML)")
            export_urdf_btn = self.server.gui.add_button("Export URDF with Spheres")

            # Status with error details (read-only)
            export_status = self.server.gui.add_markdown("Ready to export")
            export_details = self.server.gui.add_markdown("")

            @export_yml_btn.on_click
            def _(_):
                """Export sphere configuration to YAML."""
                try:
                    import yaml

                    # Create data structure for all geometry types
                    collision_geometries = {}
                    for geometry in self.geometry_store.by_id.values():
                        if geometry.link not in collision_geometries:
                            collision_geometries[geometry.link] = []

                        # Ensure clean conversion to Python primitives
                        center = geometry.local_xyz
                        if hasattr(center, "tolist"):
                            center = center.tolist()
                        else:
                            center = [float(x) for x in center]

                        # Create geometry data based on type
                        geometry_data = {
                            "center": center,
                            "type": geometry.geometry_type,
                        }

                        # Only add rotation for non-sphere geometries
                        if geometry.geometry_type != "sphere":
                            geometry_data["rpy"] = [
                                float(r) for r in geometry.local_rpy
                            ]

                        if geometry.geometry_type == "sphere":
                            geometry_data["radius"] = float(geometry.radius)
                        elif geometry.geometry_type == "box":
                            geometry_data["size"] = [float(s) for s in geometry.size]
                        elif geometry.geometry_type == "cylinder":
                            geometry_data["radius"] = float(geometry.cylinder_radius)
                            geometry_data["height"] = float(geometry.cylinder_height)

                        collision_geometries[geometry.link].append(geometry_data)

                    # Add metadata for import (ensure clean Python types)
                    data = {
                        "collision_geometries": collision_geometries,
                        # Keep backward compatibility
                        "collision_spheres": {
                            link: [g for g in geometries if g["type"] == "sphere"]
                            for link, geometries in collision_geometries.items()
                        },
                        "metadata": {
                            "total_geometries": int(len(self.geometry_store.by_id)),
                            "total_spheres": int(
                                sum(
                                    1
                                    for g in self.geometry_store.by_id.values()
                                    if g.geometry_type == "sphere"
                                )
                            ),
                            "links": list(collision_geometries.keys()),
                            "export_timestamp": float(time.time()),
                        },
                    }

                    # Determine output directory (same as URDF or current working directory)
                    if self.urdf_path and self.urdf_path.parent:
                        output_dir = self.urdf_path.parent
                    else:
                        output_dir = Path.cwd()

                    output_path = output_dir / f"{export_name_input.value}.yml"
                    output_path.write_text(
                        yaml.dump(data, default_flow_style=False, sort_keys=False)
                    )
                    export_status.content = (
                        f"✅ Exported {len(self.geometry_store.by_id)} geometries"
                    )
                    export_details.content = f"Saved to: {output_path.name}"
                    print(f"Exported spherization to {output_path.absolute()}")

                except ImportError:
                    error_msg = "PyYAML not installed. Run: pip install PyYAML"
                    export_status.content = "❌ Missing dependency"
                    export_details.content = error_msg
                    print(f"Export failed: {error_msg}")
                except Exception as e:
                    export_status.content = f"❌ Export failed: {type(e).__name__}"
                    export_details.content = str(e)
                    print(f"Export failed: {e}")

            @export_urdf_btn.on_click
            def _(_):
                """Export URDF with collision spheres."""
                try:
                    urdf_xml = inject_geometries_into_urdf_xml(
                        self.urdf_path, self.urdf, self.geometry_store
                    )

                    # Determine output directory (same as URDF or current working directory)
                    if self.urdf_path and self.urdf_path.parent:
                        output_dir = self.urdf_path.parent
                    else:
                        output_dir = Path.cwd()

                    output_path = output_dir / f"{export_name_input.value}.urdf"
                    output_path.write_text(urdf_xml)
                    export_status.content = f"✅ Exported URDF with {len(self.geometry_store.by_id)} geometries"
                    export_details.content = f"Saved to: {output_path.name}"
                    print(f"Exported spherized URDF to {output_path.absolute()}")

                except Exception as e:
                    export_status.content = f"❌ URDF export failed: {type(e).__name__}"
                    export_details.content = str(e)
                    print(f"URDF export failed: {e}")

    def _create_geometry_visualization(self, geometry: Geometry):
        """Create or update the 3D visualization for a geometry."""
        # Ensure link group exists
        if geometry.link not in self.geometry_store.group_nodes:
            # Get the link frame from enhanced URDF
            link_frame = self.urdf_viz.link_frame.get(geometry.link)
            if link_frame is not None:
                self.geometry_store.group_nodes[geometry.link] = (
                    self.server.scene.add_frame(
                        f"{link_frame.name}/geometries", show_axes=False
                    )
                )
            else:
                # Fallback: create under geometries root
                self.geometry_store.group_nodes[geometry.link] = (
                    self.server.scene.add_frame(
                        f"/geometries/{geometry.link}", show_axes=False
                    )
                )

        parent_frame = self.geometry_store.group_nodes[geometry.link]

        # Create geometry visualization with appropriate opacity
        opacity = self._get_sphere_opacity(geometry)

        # Create different geometry types
        if geometry.geometry_type == "sphere":
            # Spheres don't need rotation (they are symmetric)
            geometry.node = self.server.scene.add_icosphere(
                f"{parent_frame.name}/sphere_{geometry.id}",
                radius=geometry.radius,
                color=geometry.color,
                position=geometry.local_xyz,
                opacity=opacity,
                visible=True,
            )
        elif geometry.geometry_type == "box":
            geometry.node = self.server.scene.add_box(
                f"{parent_frame.name}/box_{geometry.id}",
                dimensions=geometry.size,
                color=geometry.color,
                position=geometry.local_xyz,
                wxyz=geometry.local_wxyz,  # Add rotation
                opacity=opacity,
                visible=True,
            )
        elif geometry.geometry_type == "cylinder":
            # Viser doesn't have add_cylinder, so we'll create a cylinder mesh using trimesh
            import trimesh
            import numpy as np

            # Create a cylinder mesh
            cylinder_mesh = trimesh.creation.cylinder(
                radius=geometry.cylinder_radius,
                height=geometry.cylinder_height,
                sections=16,  # Number of sides for the cylinder
            )

            # Set the mesh color
            # Convert RGB tuple (0-255) to RGB array (0-1)
            color_rgb = np.array(geometry.color) / 255.0
            cylinder_mesh.visual.face_colors = np.tile(
                np.append(color_rgb, opacity if opacity is not None else 1.0),
                (len(cylinder_mesh.faces), 1),
            )

            # Add the cylinder as a trimesh
            geometry.node = self.server.scene.add_mesh_trimesh(
                f"{parent_frame.name}/cylinder_{geometry.id}",
                cylinder_mesh,
                position=geometry.local_xyz,
                wxyz=geometry.local_wxyz,  # Add rotation
                visible=True,
            )

        # Make geometry clickable for selection
        @geometry.node.on_click
        def _(_):
            # Set geometry ID FIRST, before any other updates
            self.current_geometry_id = geometry.id
            self.current_sphere_id = geometry.id  # Backward compatibility
            old_link = self.current_link
            self.current_link = geometry.link

            # If we switched links, we need to update dropdowns carefully
            if old_link != self.current_link:
                self._sync_link_selection()
                # IMPORTANT: Don't call update_geometry_dropdown here as it will override our selection
                # Instead, manually update the geometry dropdown after link sync
                if self._geometry_dropdown:
                    geometries = self.geometry_store.get_geometries_for_link(
                        self.current_link
                    )
                    if geometries:
                        options = [
                            f"{g.geometry_type.title()} {g.id}" for g in geometries
                        ]
                        self._geometry_dropdown.options = options
                    self._sync_geometry_selection()
            else:
                # Same link, just update geometry selection
                self._sync_geometry_selection()

            # Update visuals and UI
            self._update_transform_control()
            self._update_radius_gizmo()
            self._update_box_resize_gizmos()
            self._update_sphere_opacities()
            self._update_mesh_visibility()
            self._update_geometry_properties_ui()

    # Keep backward compatibility alias
    def _create_sphere_visualization(self, sphere: Geometry):
        """Create or update the 3D visualization for a sphere (backward compatibility)."""
        return self._create_geometry_visualization(sphere)

    def _update_geometry_visualization(self, geometry: Geometry):
        """Update existing geometry visualization."""
        if geometry.node is not None:
            try:
                # Remove old node
                geometry.node.remove()
            except Exception:
                # Node might already be removed, ignore the error
                pass
            finally:
                # Clear the reference regardless
                geometry.node = None

        # Recreate with new properties
        self._create_geometry_visualization(geometry)

    # Keep backward compatibility alias
    def _update_sphere_visualization(self, sphere: Geometry):
        """Update existing sphere visualization (backward compatibility)."""
        return self._update_geometry_visualization(sphere)

    def _update_transform_control(self):
        """Update transform control for the currently selected sphere."""
        if (
            self.current_sphere_id is not None
            and self.current_sphere_id in self.sphere_store.by_id
        ):
            sphere = self.sphere_store.by_id[self.current_sphere_id]

            # Remove existing transform control
            self._remove_transform_control()

            # Get the parent frame for this sphere
            parent_frame = self.sphere_store.group_nodes.get(sphere.link)
            if parent_frame is not None:
                control_name = f"{parent_frame.name}/transform_control_{sphere.id}"

                # Disable rotation for spheres (they are symmetric), enable for boxes and cylinders
                disable_rotations = sphere.geometry_type == "sphere"

                self.transform_control = self.server.scene.add_transform_controls(
                    control_name,
                    scale=0.7,
                    disable_rotations=disable_rotations,
                    position=sphere.local_xyz,
                    wxyz=(
                        sphere.local_wxyz
                        if not disable_rotations
                        else (1.0, 0.0, 0.0, 0.0)
                    ),
                    depth_test=True,  # 主控制器保持深度测试
                    opacity=1.0,  # 主控制器保持完全不透明
                )

                # Set up callback for transform updates
                @self.transform_control.on_update
                def _(_):
                    if (
                        self.current_sphere_id is not None
                        and self.current_sphere_id in self.sphere_store.by_id
                    ):
                        current_sphere = self.sphere_store.by_id[self.current_sphere_id]
                        # Update position
                        current_sphere.local_xyz = tuple(
                            self.transform_control.position
                        )
                        # Update rotation only for non-sphere geometries
                        if current_sphere.geometry_type != "sphere":
                            current_sphere.update_rpy_from_quaternion(
                                tuple(self.transform_control.wxyz)
                            )
                        self._update_geometry_visualization(current_sphere)
                        self._update_radius_gizmo()
                        self._update_box_resize_gizmos()
                        # Update UI to show current rotation
                        self._update_geometry_properties_ui()

    def _remove_transform_control(self):
        """Remove the current transform control."""
        if self.transform_control is not None:
            self.transform_control.remove()
            self.transform_control = None
        self._remove_radius_gizmo()
        self._remove_box_resize_gizmos()

    def _remove_radius_gizmo(self):
        """Remove the current radius gizmo."""
        if self.radius_gizmo is not None:
            self.radius_gizmo.remove()
            self.radius_gizmo = None

    def _remove_box_resize_gizmos(self):
        """Remove all box resize gizmos."""
        for gizmo in self.box_resize_gizmos.values():
            if gizmo is not None:
                try:
                    gizmo.remove()
                except Exception:
                    pass
        self.box_resize_gizmos.clear()

    def _update_radius_gizmo(self):
        """Update radius gizmo for the currently selected sphere or cylinder."""
        # Remove any previous gizmo
        self._remove_radius_gizmo()

        if (
            self.current_sphere_id is None
            or self.current_sphere_id not in self.sphere_store.by_id
        ):
            return

        s = self.sphere_store.by_id[self.current_sphere_id]

        # Only create radius gizmo for non-box geometries (sphere and cylinder)
        if s.geometry_type == "box":
            return

        parent_frame = self.sphere_store.group_nodes.get(s.link)
        if parent_frame is None:
            return

        # Position gizmo at 135 degrees around Z-axis for better visibility
        import math

        angle = 3 * math.pi / 4  # 135 degrees

        # Get the appropriate radius based on geometry type
        if s.geometry_type == "sphere":
            current_radius = s.radius
        elif s.geometry_type == "cylinder":
            current_radius = s.cylinder_radius
        else:
            current_radius = 0.05  # fallback

        gizmo_pos = (
            s.local_xyz[0] + current_radius * math.cos(angle),  # X component at 45°
            s.local_xyz[1] + current_radius * math.sin(angle),  # Y component at 45°
            s.local_xyz[2],  # Same Z as center
        )

        gizmo_name = f"{parent_frame.name}/radius_gizmo_{s.id}"

        # Create rotation quaternion for 135 degrees around Z-axis
        # This rotates the gizmo's X-axis by 135 degrees, making it point diagonally
        from viser import transforms as tf

        rotation_135deg = tf.SO3.from_z_radians(angle)  # 135° rotation around Z

        # Create a single-axis gizmo that allows full bidirectional movement along the rotated X axis
        # This allows both increasing and decreasing radius, including going to zero
        self.radius_gizmo = self.server.scene.add_transform_controls(
            gizmo_name,
            scale=0.4,  # Reduce size to be less prominent
            active_axes=(True, False, False),  # Only X axis active (but now rotated)
            disable_sliders=True,
            disable_rotations=True,
            # Allow full range movement - no translation limits to enable zero radius
            wxyz=rotation_135deg.wxyz,  # Rotate the gizmo 135 degrees
            position=gizmo_pos,
        )

        @self.radius_gizmo.on_update
        def _(_):
            if self.current_sphere_id not in self.sphere_store.by_id:
                return

            s2 = self.sphere_store.by_id[self.current_sphere_id]

            # Only update if this is not a box geometry
            if s2.geometry_type == "box":
                return

            gizmo_pos_current = self.radius_gizmo.position

            # Calculate new radius as distance from geometry center to gizmo position
            # This is the fundamental relationship: radius = distance from center to gizmo
            center_to_gizmo = (
                gizmo_pos_current[0] - s2.local_xyz[0],
                gizmo_pos_current[1] - s2.local_xyz[1],
                gizmo_pos_current[2] - s2.local_xyz[2],
            )
            new_radius = math.sqrt(
                center_to_gizmo[0] ** 2
                + center_to_gizmo[1] ** 2
                + center_to_gizmo[2] ** 2
            )
            new_radius = max(0.005, new_radius)  # Minimum radius to avoid zero

            # Update radius based on geometry type
            if s2.geometry_type == "sphere":
                s2.radius = new_radius
                # Update UI slider without triggering callbacks
                if self._sphere_radius_slider:
                    self._updating_sphere_ui = True
                    self._sphere_radius_slider.value = new_radius
                    self._updating_sphere_ui = False
            elif s2.geometry_type == "cylinder":
                s2.cylinder_radius = new_radius
                # Update UI slider without triggering callbacks
                if self._cylinder_radius_slider:
                    self._updating_geometry_ui = True
                    self._cylinder_radius_slider.value = new_radius
                    self._updating_geometry_ui = False

            # Update visualization
            self._update_geometry_visualization(s2)

            # Don't reposition the gizmo here! Let the user drag it freely.
            # The gizmo position directly controls the radius - no secondary positioning logic needed.

    def _update_box_resize_gizmos(self):
        """Update box resize gizmos for the currently selected box geometry."""
        # Remove any previous gizmos
        self._remove_box_resize_gizmos()

        if (
            self.current_geometry_id is None
            or self.current_geometry_id not in self.geometry_store.by_id
        ):
            return

        geometry = self.geometry_store.by_id[self.current_geometry_id]

        # Only create gizmos for box geometry
        if geometry.geometry_type != "box":
            return

        parent_frame = self.geometry_store.group_nodes.get(geometry.link)
        if parent_frame is None:
            return

        import math
        import numpy as np
        from viser import transforms as tf

        # Box dimensions
        length, width, height = geometry.size
        center_x, center_y, center_z = geometry.local_xyz

        # Get the box's rotation
        try:
            # 确保quaternion是正确的格式
            wxyz = geometry.local_wxyz
            if isinstance(wxyz, tuple):
                wxyz = np.array(wxyz)
            box_rotation = tf.SO3(wxyz)
        except Exception as e:
            print(f"Error creating box rotation: {e}")
            box_rotation = tf.SO3.identity()

        # Create gizmos for each axis (X, Y, Z)
        # 负方向的gizmo轴方向会跟随box的姿态变化
        # 计算每个轴的最大向内移动距离，防止穿过中心点
        max_inward_x = -(length / 2 - 0.005)  # X轴最大向中心移动距离
        max_inward_y = -(width / 2 - 0.005)  # Y轴最大向中心移动距离
        max_inward_z = -(height / 2 - 0.005)  # Z轴最大向中心移动距离

        axes_info = [
            {
                "name": "x_neg",
                "axis": (-1, 0, 0),
                "color": (200, 80, 80),
                "local_position": np.array(
                    [-length / 2, 0, 0]
                ),  # box本地坐标系中的位置
                "active_axes": (True, False, False),  # 只允许X轴移动
                "base_rotation": tf.SO3.from_y_radians(
                    math.pi
                ),  # 基础旋转：箭头指向相反方向
                "translation_limits": (
                    (max_inward_x, 10.0),  # X轴：能向中心移动但不穿过，能向外无限移动
                    (-10.0, 10.0),  # Y轴无限制
                    (-10.0, 10.0),  # Z轴无限制
                ),
            },
            {
                "name": "y_neg",
                "axis": (0, -1, 0),
                "color": (80, 200, 80),
                "local_position": np.array([0, -width / 2, 0]),  # box本地坐标系中的位置
                "active_axes": (False, True, False),  # 只允许Y轴移动
                "base_rotation": tf.SO3.from_x_radians(
                    math.pi
                ),  # 基础旋转：箭头指向相反方向
                "translation_limits": (
                    (-10.0, 10.0),  # X轴无限制
                    (max_inward_y, 10.0),  # Y轴：能向中心移动但不穿过，能向外无限移动
                    (-10.0, 10.0),  # Z轴无限制
                ),
            },
            {
                "name": "z_neg",
                "axis": (0, 0, -1),
                "color": (80, 80, 200),
                "local_position": np.array(
                    [0, 0, -height / 2]
                ),  # box本地坐标系中的位置
                "active_axes": (False, False, True),  # 只允许Z轴移动
                "base_rotation": tf.SO3.from_x_radians(
                    math.pi
                ),  # 修复：使用base_rotation并绕X轴旋转
                "translation_limits": (
                    (-10.0, 10.0),  # X轴无限制
                    (-10.0, 10.0),  # Y轴无限制
                    (max_inward_z, 10.0),  # Z轴：能向中心移动但不穿过，能向外无限移动
                ),
            },
        ]

        for axis_info in axes_info:
            # 计算gizmo在世界坐标系中的位置
            # 将本地位置通过box的旋转变换到世界坐标系
            local_pos = axis_info["local_position"]

            # 确保local_pos是numpy数组
            if not isinstance(local_pos, np.ndarray):
                local_pos = np.array(local_pos)

            world_pos_offset = box_rotation @ local_pos  # 使用 @ 操作符进行旋转变换
            world_position = (
                center_x + world_pos_offset[0],
                center_y + world_pos_offset[1],
                center_z + world_pos_offset[2],
            )

            # 计算gizmo的旋转：box旋转 @ 基础旋转
            try:
                gizmo_rotation = box_rotation @ axis_info["base_rotation"]
            except Exception as e:
                print(f"Error computing gizmo rotation: {e}")
                gizmo_rotation = axis_info["base_rotation"]

            gizmo_name = (
                f"{parent_frame.name}/box_resize_{geometry.id}_{axis_info['name']}"
            )

            # 获取translation_limits参数
            translation_limits = axis_info.get(
                "translation_limits",
                ((-1000.0, 1000.0), (-1000.0, 1000.0), (-1000.0, 1000.0)),
            )

            gizmo = self.server.scene.add_transform_controls(
                gizmo_name,
                scale=0.2,
                line_width=20.0,
                active_axes=axis_info["active_axes"],
                disable_sliders=True,
                disable_rotations=True,
                wxyz=gizmo_rotation.wxyz,
                position=world_position,
                opacity=0.8,
                depth_test=False,
                translation_limits=translation_limits,  # 添加移动限制
            )

            # Store the gizmo and setup callback
            self.box_resize_gizmos[axis_info["name"]] = gizmo

            # Create callback for this specific gizmo
            self._setup_box_resize_callback(gizmo, axis_info["name"], axis_info["axis"])

    def _setup_box_resize_callback(self, gizmo, axis_name, axis_direction):
        """Setup callback for a specific box resize gizmo."""

        @gizmo.on_update
        def _(_):
            if (
                self.current_geometry_id is None
                or self.current_geometry_id not in self.geometry_store.by_id
            ):
                return

            geometry = self.geometry_store.by_id[self.current_geometry_id]

            # Only update if this is a box geometry
            if geometry.geometry_type != "box":
                return

            # Get current gizmo position
            gizmo_pos = gizmo.position
            center = geometry.local_xyz
            old_size = geometry.size

            # 计算考虑旋转的新尺寸
            import numpy as np
            from viser import transforms as tf

            # 获取box的旋转
            try:
                wxyz = geometry.local_wxyz
                if isinstance(wxyz, tuple):
                    wxyz = np.array(wxyz)
                box_rotation = tf.SO3(wxyz)
                # 计算逆旋转，用于将世界坐标转换回box局部坐标
                box_rotation_inv = tf.SO3.from_matrix(
                    np.linalg.inv(box_rotation.as_matrix())
                )
            except Exception as e:
                print(f"Error computing box rotation: {e}")
                box_rotation = tf.SO3.identity()
                box_rotation_inv = tf.SO3.identity()

            # 计算gizmo位置在box局部坐标系中的位置
            gizmo_pos_local = np.array(gizmo_pos) - np.array(center)  # 相对于中心的位置
            try:
                # 将世界坐标转换回box局部坐标
                gizmo_pos_local = box_rotation_inv @ gizmo_pos_local
            except Exception as e:
                print(f"Error transforming coordinates: {e}")

            # 计算新尺寸
            new_size = list(geometry.size)

            if "x" in axis_name:
                # X-axis resize - 使用局部坐标系中的X轴距离
                distance = abs(gizmo_pos_local[0])
                new_size[0] = max(0.01, distance * 2)  # Minimum size 1cm
            elif "y" in axis_name:
                # Y-axis resize - 使用局部坐标系中的Y轴距离
                distance = abs(gizmo_pos_local[1])
                new_size[1] = max(0.01, distance * 2)  # Minimum size 1cm
            elif "z" in axis_name:
                # Z-axis resize - 使用局部坐标系中的Z轴距离
                distance = abs(gizmo_pos_local[2])
                new_size[2] = max(0.01, distance * 2)  # Minimum size 1cm

            # Update geometry size
            geometry.size = tuple(new_size)

            # Update visualization
            self._update_geometry_visualization(geometry)

            # Update other gizmos positions (but avoid infinite recursion)
            self._update_other_box_gizmos(axis_name, geometry)

            # Update UI sliders without triggering callbacks
            if self._box_size_sliders:
                self._updating_geometry_ui = True
                self._box_size_sliders[0].value = new_size[0]  # length
                self._box_size_sliders[1].value = new_size[1]  # width
                self._box_size_sliders[2].value = new_size[2]  # height
                self._updating_geometry_ui = False

    def _update_other_box_gizmos(self, changed_axis_name, geometry):
        """Update positions of other box gizmos when one is moved."""

        # Only update if this is a box geometry
        if geometry.geometry_type != "box":
            return

        import math
        import numpy as np
        from viser import transforms as tf

        length, width, height = geometry.size
        center_x, center_y, center_z = geometry.local_xyz

        # Get the box's rotation
        try:
            # 确保quaternion是正确的格式
            wxyz = geometry.local_wxyz
            if isinstance(wxyz, tuple):
                wxyz = np.array(wxyz)
            box_rotation = tf.SO3(wxyz)
        except Exception as e:
            print(f"Error creating box rotation: {e}")
            box_rotation = tf.SO3.identity()

        # Update positions of gizmos that weren't just moved
        local_positions = {
            "x_neg": np.array([-length / 2, 0, 0]),
            "y_neg": np.array([0, -width / 2, 0]),
            "z_neg": np.array([0, 0, -height / 2]),
        }

        for axis_name, gizmo in self.box_resize_gizmos.items():
            if axis_name != changed_axis_name and gizmo is not None:
                try:
                    # 计算新的世界坐标位置，考虑box的旋转
                    local_pos = local_positions[axis_name]
                    world_pos_offset = box_rotation @ local_pos  # 使用 @ 操作符
                    world_position = (
                        center_x + world_pos_offset[0],
                        center_y + world_pos_offset[1],
                        center_z + world_pos_offset[2],
                    )
                    gizmo.position = world_position
                except Exception:
                    pass  # Ignore errors if gizmo is being removed

    def _update_geometry_properties_ui(self):
        """Update the geometry property UI controls to reflect the currently selected geometry."""
        # Set flag to prevent recursive updates
        self._updating_geometry_ui = True
        self._updating_sphere_ui = True  # For backward compatibility

        if (
            self.current_geometry_id is not None
            and self.current_geometry_id in self.geometry_store.by_id
        ):
            geometry = self.geometry_store.by_id[self.current_geometry_id]

            # Update properties based on geometry type
            if geometry.geometry_type == "sphere":
                if self._sphere_radius_slider:
                    self._sphere_radius_slider.value = geometry.radius
            elif geometry.geometry_type == "box":
                if self._box_size_sliders:
                    self._box_size_sliders[0].value = geometry.size[0]  # length
                    self._box_size_sliders[1].value = geometry.size[1]  # width
                    self._box_size_sliders[2].value = geometry.size[2]  # height
            elif geometry.geometry_type == "cylinder":
                if self._cylinder_radius_slider:
                    self._cylinder_radius_slider.value = geometry.cylinder_radius
                if self._cylinder_height_slider:
                    self._cylinder_height_slider.value = geometry.cylinder_height

            # Update color input
            if self._geometry_color_input:
                self._geometry_color_input.value = geometry.color

            # Update rotation sliders only for non-sphere geometries
            if self._rpy_sliders:
                if geometry.geometry_type != "sphere":
                    self._rpy_sliders[0].value = geometry.local_rpy[0]  # roll
                    self._rpy_sliders[1].value = geometry.local_rpy[1]  # pitch
                    self._rpy_sliders[2].value = geometry.local_rpy[2]  # yaw
                    # Enable rotation sliders
                    self._rpy_sliders[0].disabled = False
                    self._rpy_sliders[1].disabled = False
                    self._rpy_sliders[2].disabled = False
                else:
                    # For spheres, reset to zero and disable sliders
                    self._rpy_sliders[0].value = 0.0
                    self._rpy_sliders[1].value = 0.0
                    self._rpy_sliders[2].value = 0.0
                    self._rpy_sliders[0].disabled = True
                    self._rpy_sliders[1].disabled = True
                    self._rpy_sliders[2].disabled = True
        else:
            # Reset to default values when no geometry selected
            if self._sphere_radius_slider:
                self._sphere_radius_slider.value = 0.05
            if self._box_size_sliders:
                self._box_size_sliders[0].value = 0.1
                self._box_size_sliders[1].value = 0.1
                self._box_size_sliders[2].value = 0.1
            if self._cylinder_radius_slider:
                self._cylinder_radius_slider.value = 0.05
            if self._cylinder_height_slider:
                self._cylinder_height_slider.value = 0.1
            if self._geometry_color_input:
                self._geometry_color_input.value = (255, 180, 60)
            if self._rpy_sliders:
                self._rpy_sliders[0].value = 0.0  # roll
                self._rpy_sliders[1].value = 0.0  # pitch
                self._rpy_sliders[2].value = 0.0  # yaw
                # Disable rotation sliders when no geometry is selected
                self._rpy_sliders[0].disabled = True
                self._rpy_sliders[1].disabled = True
                self._rpy_sliders[2].disabled = True

        # Clear flag after UI update
        self._updating_geometry_ui = False
        self._updating_sphere_ui = False

    # Keep backward compatibility alias
    def _update_sphere_properties_ui(self):
        """Update the sphere property UI controls (backward compatibility)."""
        return self._update_geometry_properties_ui()

    def _sync_link_selection(self):
        """Sync link selection between visibility controls and sphere editor."""
        # Sync visibility dropdown if different
        if (
            self._current_link_dropdown
            and self._current_link_dropdown.value != self.current_link
        ):
            self._current_link_dropdown.value = self.current_link
        # Sync sphere editor dropdown if different
        if self._link_dropdown and self._link_dropdown.value != self.current_link:
            self._link_dropdown.value = self.current_link

    def _sync_geometry_selection(self):
        """Sync geometry dropdown to reflect the currently selected geometry."""
        if self._geometry_dropdown and self.current_geometry_id is not None:
            # Find the correct dropdown option for this geometry
            if self.current_geometry_id in self.geometry_store.by_id:
                geometry = self.geometry_store.by_id[self.current_geometry_id]
                expected_value = f"{geometry.geometry_type.title()} {geometry.id}"

                # Check if this value exists in the dropdown options
                if expected_value in self._geometry_dropdown.options:
                    self._geometry_dropdown.value = expected_value

    # Keep backward compatibility alias
    def _sync_sphere_selection(self):
        """Sync sphere dropdown to reflect the currently selected sphere (backward compatibility)."""
        return self._sync_geometry_selection()

    def _get_sphere_opacity(self, sphere: Geometry) -> float:
        """Get the appropriate opacity for a sphere based on current selection state."""
        if sphere.id == self.current_sphere_id:
            return self.selected_sphere_opacity
        elif sphere.link == self.current_link:
            return self.unselected_spheres_opacity
        else:
            return self.other_links_spheres_opacity

    def _update_mesh_visibility(self):
        """Update visibility of robot meshes based on link selection."""
        for link_name, mesh_handles in self.urdf_viz.link_meshes.items():
            for mesh_handle in mesh_handles:
                # Determine if this link should be visible
                if link_name == self.current_link:
                    # This is the selected link
                    mesh_handle.visible = self.show_selected_link
                else:
                    # This is a non-selected link
                    mesh_handle.visible = self.show_other_links

    def _update_sphere_opacities(self):
        """Update opacity of all spheres based on current selection state."""
        for sphere in self.sphere_store.by_id.values():
            if sphere.node is not None:
                new_opacity = self._get_sphere_opacity(sphere)
                # Update sphere opacity
                sphere.node.opacity = new_opacity
                # Handle visibility (0.0 opacity = invisible)
                sphere.node.visible = new_opacity > 0.0

    def _load_spherization_yaml(self, yaml_path: Path):
        """Load sphere configuration from YAML file at startup."""
        try:
            import yaml

            if not yaml_path.exists():
                print(f"⚠️  Spherization YAML file not found: {yaml_path}")
                return

            print(f"📥 Loading spherization from: {yaml_path}")
            data = yaml.safe_load(yaml_path.read_text())
            collision_spheres = data.get("collision_spheres", {})

            # Import spheres and geometries
            total_loaded = 0

            # Try to load new format with collision_geometries first
            collision_data = data.get("collision_geometries", {})
            if collision_data:
                for link_name, geometries_data in collision_data.items():
                    for geom_data in geometries_data:
                        # Extract rotation if available
                        rpy = tuple(geom_data.get("rpy", [0.0, 0.0, 0.0]))

                        # Determine geometry type and parameters
                        geom_type = geom_data.get("type", "sphere")

                        if geom_type == "sphere":
                            geometry = self.geometry_store.add(
                                link_name,
                                xyz=tuple(geom_data["center"]),
                                geometry_type="sphere",
                                radius=geom_data["radius"],
                                rpy=rpy,
                            )
                        elif geom_type == "box":
                            geometry = self.geometry_store.add(
                                link_name,
                                xyz=tuple(geom_data["center"]),
                                geometry_type="box",
                                size=tuple(geom_data["size"]),
                                rpy=rpy,
                            )
                        elif geom_type == "cylinder":
                            geometry = self.geometry_store.add(
                                link_name,
                                xyz=tuple(geom_data["center"]),
                                geometry_type="cylinder",
                                cylinder_radius=geom_data["radius"],
                                cylinder_height=geom_data["height"],
                                rpy=rpy,
                            )

                        self._create_geometry_visualization(geometry)
                        total_loaded += 1
            else:
                # Fallback to old format with collision_spheres
                for link_name, spheres_data in collision_spheres.items():
                    for sphere_data in spheres_data:
                        # Old format - only spheres, no rotation
                        sphere = self.sphere_store.add(
                            link_name,
                            xyz=tuple(sphere_data["center"]),
                            radius=sphere_data["radius"],
                        )
                        self._create_geometry_visualization(sphere)
                        total_loaded += 1

            print(f"✅ Loaded {total_loaded} spheres from {yaml_path.name}")

        except ImportError:
            print("⚠️  PyYAML not installed. Cannot load spherization YAML.")
            print("   Install with: pip install PyYAML")
        except Exception as e:
            print(f"❌ Failed to load spherization YAML: {e}")

    def _add_reference_grid(self):
        """Add a reference grid to the scene."""
        # Get scene bounds to position grid appropriately
        try:
            trimesh_scene = (
                self.urdf_viz._urdf.scene or self.urdf_viz._urdf.collision_scene
            )
            z_pos = trimesh_scene.bounds[0, 2] if trimesh_scene is not None else 0.0
        except:
            z_pos = 0.0

        self.server.scene.add_grid(
            "/reference_grid",
            width=2,
            height=2,
            position=(0.0, 0.0, z_pos),
            cell_color=(200, 200, 200),
            cell_thickness=1.0,
        )

    def run(self):
        """Run the application (blocking)."""
        print("🚀 Application running! Use Ctrl+C to exit.")
        try:
            while True:
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("\n👋 Shutting down Bubblify...")
        finally:
            # Cleanup
            self._remove_transform_control()
            self._remove_radius_gizmo()
            self.urdf_viz.remove()
