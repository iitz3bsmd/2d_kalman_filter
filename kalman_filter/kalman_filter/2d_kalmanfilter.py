import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
import numpy as np
from numpy.linalg import inv

class KalmanFilter(Node):
    def __init__(self):
        super().__init__('kalman_filter_node')
        # Initialize kalman variables
        self.dt = 0.001
        self.vel = np.zeros((2,1))
        self.state = np.zeros((2,1))
        self.measurement = np.zeros((2,1))
        self.vel_cov = 5*self.dt*np.eye(2)
        self.state_cov = np.eye(2)
        self.measurement_cov = 0.1*np.eye(2)
        self.state_transform = np.eye(2)*self.dt
        self.k = 0

        # Subscribe to the /odom_noise topic
        self.subscription = self.create_subscription(Odometry,
                                                     '/odom_noise',
                                                     self.odom_callback,
                                                     1)
        
        #publish the estimated reading
        self.estimated_pub=self.create_publisher(Odometry,
                                                 "/odom_estimated",1)

    def odom_callback(self, msg):
        # Extract the position measurements from the Odometry message
        self.measurement[0] = msg.pose.pose.position.x
        self.measurement[1] = msg.pose.pose.position.y
        self.vel[0] = msg.twist.twist.linear.x
        self.vel[1] = msg.twist.twist.linear.y

        # Prediction step
        self.state = self.state + np.dot(self.state_transform, self.vel)
        self.state_cov = self.state_cov + self.vel_cov

        # Update step
        tmp = self.state_cov + self.measurement_cov
        self.k = np.dot(self.state_cov, np.linalg.inv(tmp))

        tmp = self.measurement - self.state
        self.state = self.state + np.dot(self.k, tmp)

        tmp = np.eye(2) - self.k
        self.state_cov = np.dot(tmp, self.state_cov)

        #publish the estimated reading
        msg.pose.pose.position.x = float(self.state[0])
        msg.pose.pose.position.y = float(self.state[1])
        self.estimated_pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = KalmanFilter()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
