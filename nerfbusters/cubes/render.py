"""Utils for rendering meshes.
"""
import os
os.environ['PYOPENGL_PLATFORM'] = 'egl'
import copy
import numpy as np

def render_scenemesh(obj_mesh_data, pose_in, camera, W, H, alphaMode="OPAQUE", color=(0.5, 0.5, 0.5, 1.0)):
    """
    obj_meshes - obj_mesh or list of obj_meshes
    """
    pose = copy.deepcopy(pose_in)
    if pose.shape == (3, 4):
        pose = np.concatenate([pose, np.zeros_like(pose[0:1])], axis=0)
        pose[3, 3] = 1.0

    # bg_color is important for the alpha mask to work properly
    import pyrender
    scene = pyrender.Scene(ambient_light=(0.5, 0.5, 0.5),
                           bg_color=(1.0, 1.0, 1.0, 0.0))

    if not isinstance(obj_mesh_data, list):
        obj_mesh_data = [obj_mesh_data]

    for i in range(len(obj_mesh_data)):
        obj_mesh = obj_mesh_data[i]
        material = pyrender.MetallicRoughnessMaterial(
            metallicFactor=0.2,
            alphaMode=alphaMode,
            baseColorFactor=color)

        mesh = pyrender.Mesh.from_trimesh(obj_mesh, material=material)
        scene.add(mesh)

    scene.add(camera, pose=pose)

    light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=1)
    light_pose = copy.deepcopy(pose)

    light_pose[:3, 3] = np.array([0, -1, 1])
    scene.add(light, pose=light_pose)

    light_pose[:3, 3] = np.array([0, 1, 1])
    scene.add(light, pose=light_pose)

    light_pose[:3, 3] = np.array([1, 1, 2])
    scene.add(light, pose=light_pose)

    r = pyrender.OffscreenRenderer(W, H)
    image_alpha, depth = r.render(scene, flags=pyrender.RenderFlags.RGBA)
    alpha = image_alpha[:, :, 3].astype(np.float32) / 255.0
    image = image_alpha[:, :, :3]
    r.delete()
    return image, depth, alpha
