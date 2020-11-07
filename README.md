# Human Skin Segmentation
Person segmentation (DeepLabV3 from torchvision) and Skin Segmentation using OpenCV Color Range masking

# How to use
1. Run on a single image `python skin_analyzer_folder.py`
```python
    ps = PersonSegmentation('cpu', is_resize=True, resize_size=480)
    seg_map = ps.person_segment("image.jpg")
    frame = ps.decode_segmap(seg_map, "image.jpg")
    skin_frame, skin2img_ratio = ps.skin_segment_pro(frame)
```


#### Contributing to this repo
[How to Contribute](CONTRIBUTING.md)

#### References
1. [skin detection method 2](https://github.com/Jeanvit/PySkinDetection)

#### Gotchas
1. Inverse normalize + convert2bgr from torch tensor to opencv matrix is non equivalent to direct `cv2.imread`
```python
    input_frame = self.inverse_normalize(self.x, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    frame = input_frame[0].permute(1, 2, 0).numpy()
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
```