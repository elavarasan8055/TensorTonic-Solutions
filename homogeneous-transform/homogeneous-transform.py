import numpy as np

def apply_homogeneous_transform(T, points):
    """
    Apply 4x4 homogeneous transform T to 3D point(s).
    """
    # Your code here
    T=np.array(T)
    points=np.array(points)
    ones=np.ones((points.shape[0],1))
    print(points.ndim)
    if points.ndim==1:
        points = np.append(points, 1)
        points=points.reshape(1,-1)
    else:
        points=np.append(points,ones,axis=1)
    transformed_points= np.dot(T,points.T).T
    return transformed_points[:,:3] if transformed_points.shape[0]>1 else transformed_points[0][:3]
  