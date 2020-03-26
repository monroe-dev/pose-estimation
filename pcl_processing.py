import pyrealsense2 as rs
import open3d as o3d


def makePly(file_name):
    # Declare pointcloud object, for calculating pointclouds and texture mappings
    pc = rs.pointcloud()
    # We want the points object to be persistent so we can display the last cloud when a frame drops
    points = rs.points()

    # Declare RealSense pipeline, encapsulating the actual device and sensors
    pipe = rs.pipeline()
    config = rs.config()
    # Enable depth stream
    config.enable_stream(rs.stream.depth)

    # Start streaming with chosen configuration
    pipe.start(config)

    # We'll use the colorizer to generate texture for our PLY
    # (alternatively, texture can be obtained from color or infrared stream)
    colorizer = rs.colorizer()

    try:
        # Wait for the next set of frames from the camera
        frames = pipe.wait_for_frames()
        colorized = colorizer.process(frames)

        # Create save_to_ply object
        ply = rs.save_to_ply(file_name)

        # Set options to the desired values
        # In this example we'll generate a textual PLY with normals (mesh is already created by default)
        # ply.set_option(rs.save_to_ply.option_ply_binary, False)
        # ply.set_option(rs.save_to_ply.option_ply_normals, True)

        print("Saving to", file_name, "...")
        # Apply the processing block to the frameset which contains the depth frame and the texture
        ply.process(colorized)
        print("Done")
    finally:
        pipe.stop()


def detectPlane(file_name, distance, n, num):
    pcd = o3d.io.read_point_cloud(file_name)

    print("Find the plane model and the inliers of the largest planar segment in the point cloud.")
    plane_model, inliers = pcd.segment_plane(distance_threshold=distance,
                                             ransac_n=n,
                                             num_iterations=num)
    [a, b, c, d] = plane_model
    print(f"Plane model: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")
    inlier_cloud = pcd.select_down_sample(inliers)
    inlier_cloud.paint_uniform_color([1.0, 0, 0])
    outlier_cloud = pcd.select_down_sample(inliers, invert=True)
    o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])


if __name__ == '__main__':
    # makePly('data/original.ply')
    file_name = 'data/crop.ply'
    dist_threshold = 0.005
    ransac_n = 3
    iteration = 250
    detectPlane(file_name, dist_threshold, ransac_n, iteration)