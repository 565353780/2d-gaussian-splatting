from copy import deepcopy

from typing import List

from camera_control.Module.camera import Camera

from twod_gs.Module.gs_renderer import GSRenderer


def render_cameras(
    gs_ply_file_path: str,
    camera_list: List[Camera],
    sh_degree: int = 3,
    bg_color: list = [1, 1, 1],
    device: str = 'cuda:0',
) -> List[Camera]:
    """Render 2DGS images + depths into a fresh copy of the cameras.

    输入的 ``camera_list`` 不会被修改：函数内部先 ``deepcopy`` 出一份
    ``rendered_camera_list``，把每个相机的渲染图、深度写回该副本并把
    ``mask`` 置空后返回。该接口与 ``fast_gs.API.renderer.render_cameras``
    保持同签名、同语义。
    """
    rendered_camera_list = deepcopy(camera_list)

    render_list = GSRenderer.renderCameras(
        gs_ply_file_path,
        rendered_camera_list,
        sh_degree=sh_degree,
        bg_color=bg_color,
        device=device,
    )

    for i, rendered_camera in enumerate(rendered_camera_list):
        image = render_list[i]['render'].permute(1, 2, 0)
        depth = render_list[i]['rend_depth']

        rendered_camera.loadImage(image)
        rendered_camera.loadDepth(depth)

        rendered_camera.mask = None

    return rendered_camera_list
