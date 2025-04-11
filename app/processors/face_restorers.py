from typing import TYPE_CHECKING

import torch
import numpy as np
from torchvision.transforms import v2
from skimage import transform as trans

if TYPE_CHECKING:
    from app.processors.models_processor import ModelsProcessor

class FaceRestorers:
    def __init__(self, models_processor: 'ModelsProcessor'):
        self.models_processor = models_processor

    def apply_facerestorer(self, swapped_face_upscaled, restorer_det_type, restorer_type, restorer_blend, fidelity_weight, detect_score):
        temp = swapped_face_upscaled
        t512 = v2.Resize((512, 512), antialias=False)
        t256 = v2.Resize((256, 256), antialias=False)
        t1024 = v2.Resize((1024, 1024), antialias=False)
        t2048 = v2.Resize((2048, 2048), antialias=False)

        # 如果使用单独的检测模式
        if restorer_det_type == 'Blend' or restorer_det_type == 'Reference':
            if restorer_det_type == 'Blend':
                # 设置变换
                dst = self.models_processor.arcface_dst * 4.0
                dst[:,0] += 32.0

            elif restorer_det_type == 'Reference':
                try:
                    dst, _, _ = self.models_processor.run_detect_landmark(swapped_face_upscaled, bbox=np.array([0, 0, 512, 512]), det_kpss=[], detect_mode='5', score=detect_score/100.0, from_points=False)
                except Exception as e: # pylint: disable=broad-except
                    print(f"检测异常: {e}")
                    return swapped_face_upscaled

            # 如果关键点为空，返回未增强的人脸
            if not isinstance(dst, np.ndarray) or len(dst)==0:
                return swapped_face_upscaled
            
            tform = trans.SimilarityTransform()
            try:
                tform.estimate(dst, self.models_processor.FFHQ_kps)
            except:
                return swapped_face_upscaled
            # 变换、缩放和归一化
            temp = v2.functional.affine(swapped_face_upscaled, tform.rotation*57.2958, (tform.translation[0], tform.translation[1]) , tform.scale, 0, center = (0,0) )
            temp = v2.functional.crop(temp, 0,0, 512, 512)

        temp = torch.div(temp, 255)
        temp = v2.functional.normalize(temp, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=False)

        if restorer_type == 'GPEN-256':
            temp = t256(temp)

        temp = torch.unsqueeze(temp, 0).contiguous()

        # 绑定
        outpred = torch.empty((1,3,512,512), dtype=torch.float32, device=self.models_processor.device).contiguous()

        if restorer_type == 'GFPGAN-v1.4':
            self.run_GFPGAN(temp, outpred)

        elif restorer_type == 'CodeFormer':
            self.run_codeformer(temp, outpred, fidelity_weight)

        elif restorer_type == 'GPEN-256':
            outpred = torch.empty((1,3,256,256), dtype=torch.float32, device=self.models_processor.device).contiguous()
            self.run_GPEN_256(temp, outpred)

        elif restorer_type == 'GPEN-512':
            self.run_GPEN_512(temp, outpred)

        elif restorer_type == 'GPEN-1024':
            temp = t1024(temp)
            outpred = torch.empty((1, 3, 1024, 1024), dtype=torch.float32, device=self.models_processor.device).contiguous()
            self.run_GPEN_1024(temp, outpred)

        elif restorer_type == 'GPEN-2048':
            temp = t2048(temp)
            outpred = torch.empty((1, 3, 2048, 2048), dtype=torch.float32, device=self.models_processor.device).contiguous()
            self.run_GPEN_2048(temp, outpred)

        elif restorer_type == 'RestoreFormer++':
            self.run_RestoreFormerPlusPlus(temp, outpred)

        elif restorer_type == 'VQFR-v2':
            self.run_VQFR_v2(temp, outpred, fidelity_weight)

        # 格式化回 cxHxW @ 255
        outpred = torch.squeeze(outpred)
        outpred = torch.clamp(outpred, -1, 1)
        outpred = torch.add(outpred, 1)
        outpred = torch.div(outpred, 2)
        outpred = torch.mul(outpred, 255)

        if restorer_type == 'GPEN-256' or restorer_type == 'GPEN-1024' or restorer_type == 'GPEN-2048':
            outpred = t512(outpred)

        # 反转变换
        if restorer_det_type == 'Blend' or restorer_det_type == 'Reference':
            outpred = v2.functional.affine(outpred, tform.inverse.rotation*57.2958, (tform.inverse.translation[0], tform.inverse.translation[1]), tform.inverse.scale, 0, interpolation=v2.InterpolationMode.BILINEAR, center = (0,0) )

        # 混合
        alpha = float(restorer_blend)/100.0
        outpred = torch.add(torch.mul(outpred, alpha), torch.mul(swapped_face_upscaled, 1-alpha))

        self.sync()
        return outpred

    def run_GFPGAN(self, image, output):
        if not self.models_processor.models['GFPGANv1.4']:
            self.models_processor.models['GFPGANv1.4'] = self.models_processor.load_model('GFPGANv1.4')

        io_binding = self.models_processor.models['GFPGANv1.4'].io_binding()
        io_binding.bind_input(name='input', device_type=self.models_processor.device, device_id=0, element_type=np.float32, shape=(1,3,512,512), buffer_ptr=image.data_ptr())
        io_binding.bind_output(name='output', device_type=self.models_processor.device, device_id=0, element_type=np.float32, shape=(1,3,512,512), buffer_ptr=output.data_ptr())

        self.sync()
        self.models_processor.models['GFPGANv1.4'].run_with_iobinding(io_binding)

    def run_GPEN_256(self, image, output):
        if not self.models_processor.models['GPENBFR256']:
            self.models_processor.models['GPENBFR256'] = self.models_processor.load_model('GPENBFR256')

        io_binding = self.models_processor.models['GPENBFR256'].io_binding()
        io_binding.bind_input(name='input', device_type=self.models_processor.device, device_id=0, element_type=np.float32, shape=(1,3,256,256), buffer_ptr=image.data_ptr())
        io_binding.bind_output(name='output', device_type=self.models_processor.device, device_id=0, element_type=np.float32, shape=(1,3,256,256), buffer_ptr=output.data_ptr())

        self.sync()
        self.models_processor.models['GPENBFR256'].run_with_iobinding(io_binding)

    def run_GPEN_512(self, image, output):
        if not self.models_processor.models['GPENBFR512']:
            self.models_processor.models['GPENBFR512'] = self.models_processor.load_model('GPENBFR512')

        io_binding = self.models_processor.models['GPENBFR512'].io_binding()
        io_binding.bind_input(name='input', device_type=self.models_processor.device, device_id=0, element_type=np.float32, shape=(1,3,512,512), buffer_ptr=image.data_ptr())
        io_binding.bind_output(name='output', device_type=self.models_processor.device, device_id=0, element_type=np.float32, shape=(1,3,512,512), buffer_ptr=output.data_ptr())

        self.sync()
        self.models_processor.models['GPENBFR512'].run_with_iobinding(io_binding)

    def run_GPEN_1024(self, image, output):
        if not self.models_processor.models['GPENBFR1024']:
            self.models_processor.models['GPENBFR1024'] = self.models_processor.load_model('GPENBFR1024')

        io_binding = self.models_processor.models['GPENBFR1024'].io_binding()
        io_binding.bind_input(name='input', device_type=self.models_processor.device, device_id=0, element_type=np.float32, shape=(1,3,1024,1024), buffer_ptr=image.data_ptr())
        io_binding.bind_output(name='output', device_type=self.models_processor.device, device_id=0, element_type=np.float32, shape=(1,3,1024,1024), buffer_ptr=output.data_ptr())

        self.sync()
        self.models_processor.models['GPENBFR1024'].run_with_iobinding(io_binding)

    def run_GPEN_2048(self, image, output):
        if not self.models_processor.models['GPENBFR2048']:
            self.models_processor.models['GPENBFR2048'] = self.models_processor.load_model('GPENBFR2048')

        io_binding = self.models_processor.models['GPENBFR2048'].io_binding()
        io_binding.bind_input(name='input', device_type=self.models_processor.device, device_id=0, element_type=np.float32, shape=(1,3,2048,2048), buffer_ptr=image.data_ptr())
        io_binding.bind_output(name='output', device_type=self.models_processor.device, device_id=0, element_type=np.float32, shape=(1,3,2048,2048), buffer_ptr=output.data_ptr())

        self.sync()
        self.models_processor.models['GPENBFR2048'].run_with_iobinding(io_binding)

    def run_codeformer(self, image, output, fidelity_weight_value=0.9):
        if not self.models_processor.models['codeformer']:
            self.models_processor.models['codeformer'] = self.models_processor.load_model('codeformer')

        io_binding = self.models_processor.models['codeformer'].io_binding()
        io_binding.bind_input(name='input', device_type=self.models_processor.device, device_id=0, element_type=np.float32, shape=(1,3,512,512), buffer_ptr=image.data_ptr())
        io_binding.bind_input(name='fidelity_weight', device_type=self.models_processor.device, device_id=0, element_type=np.float32, shape=(1,), buffer_ptr=np.array([fidelity_weight_value], dtype=np.float32).data_ptr())
        io_binding.bind_output(name='output', device_type=self.models_processor.device, device_id=0, element_type=np.float32, shape=(1,3,512,512), buffer_ptr=output.data_ptr())

        self.sync()
        self.models_processor.models['codeformer'].run_with_iobinding(io_binding)

    def run_VQFR_v2(self, image, output, fidelity_ratio_value):
        if not self.models_processor.models['vqfr_v2']:
            self.models_processor.models['vqfr_v2'] = self.models_processor.load_model('vqfr_v2')

        io_binding = self.models_processor.models['vqfr_v2'].io_binding()
        io_binding.bind_input(name='input', device_type=self.models_processor.device, device_id=0, element_type=np.float32, shape=(1,3,512,512), buffer_ptr=image.data_ptr())
        io_binding.bind_input(name='fidelity_ratio', device_type=self.models_processor.device, device_id=0, element_type=np.float32, shape=(1,), buffer_ptr=np.array([fidelity_ratio_value], dtype=np.float32).data_ptr())
        io_binding.bind_output(name='output', device_type=self.models_processor.device, device_id=0, element_type=np.float32, shape=(1,3,512,512), buffer_ptr=output.data_ptr())

        self.sync()
        self.models_processor.models['vqfr_v2'].run_with_iobinding(io_binding)

    def run_RestoreFormerPlusPlus(self, image, output):
        if not self.models_processor.models['restoreformer_plus_plus']:
            self.models_processor.models['restoreformer_plus_plus'] = self.models_processor.load_model('restoreformer_plus_plus')

        io_binding = self.models_processor.models['restoreformer_plus_plus'].io_binding()
        io_binding.bind_input(name='input', device_type=self.models_processor.device, device_id=0, element_type=np.float32, shape=(1,3,512,512), buffer_ptr=image.data_ptr())
        io_binding.bind_output(name='output', device_type=self.models_processor.device, device_id=0, element_type=np.float32, shape=(1,3,512,512), buffer_ptr=output.data_ptr())

        self.sync()
        self.models_processor.models['restoreformer_plus_plus'].run_with_iobinding(io_binding)

    def sync(self):
        if self.models_processor.device == 'cuda':
            torch.cuda.synchronize()
        elif self.models_processor.device == 'mps':
            torch.mps.synchronize()
        elif self.models_processor.device != 'cpu':
            self.models_processor.syncvec.cpu()