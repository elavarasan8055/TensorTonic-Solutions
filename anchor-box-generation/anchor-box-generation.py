def generate_anchors(feature_size, image_size, scales, aspect_ratios):
    """
    Generate anchor boxes for object detection.
    """
    # Write code here
    stride_area=image_size**2/feature_size**2
    stride=int(stride_area**0.5)
    

    cols=image_size//stride
    step_size=image_size//cols
    centers=[[i*step_size+step_size*0.5,j*step_size+step_size*0.5] for i in range(cols) for j in range(cols)]

     
    anchors=[[center[1]-(scale*ar**0.5)/2  ,center[0]-(scale/ar**0.5)/2,center[1]+(scale*ar**0.5)/2,center[0]+(scale/ar**0.5)/2] for center in centers for scale in scales for ar in aspect_ratios]
    
    return anchors