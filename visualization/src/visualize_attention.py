import threading
import time

import numpy as np
from scipy.ndimage.filters import gaussian_filter

from bokeh.client import push_session
from bokeh.layouts import column, row
from bokeh.models import LinearColorMapper, ColumnDataSource, HoverTool
from bokeh.palettes import brewer
import bokeh.plotting as bkp

class VisualizeAttention():
    """This class takes attention data an shows it in a bokeh plot"""
    def __init__(self):

        # General settings
        self.n_scans = 540                  # Number of scan displayed (can be less than real scans to improve performance
        self.n_feature = 40                 # Number of features (columns)
        self.n_filter = 64                  # Number of filters (rows)
        self.graph_update_period = 1.0/5.0  # Redraw the plots with this period
        self.attention_scale_max = 0.03     # Maximum value for the ploted range
        self.plot_attention_matrix = False  # Select to show the map or the matrix

        # Data captures for threading
        self.laser_data = np.zeros([self.n_scans])
        self.target_pose = np.zeros([3])
        self.sensor_attention_data = np.ones([self.n_filter,self.n_feature])

        # Create the initiale plots
        self.session = push_session(bkp.curdoc())
        self.create_plot()
        self.session.show(self.plots)

        # Use a separate thread to process the received data
        self.interrupt_event = threading.Event()
        self.processing_thread = threading.Thread(target=self.process_data)
        self.processing_thread.start()

    def close(self):
        self.interrupt_event.set()
        self.processing_thread.join()

    def process_data(self):
        """
        This is the thread function to decouple updates in the ROS messages from the plotting
        process. Plots are redrawn with a fixed frequency and not on every single data update.
        """
        
        next_call = time.time()
        while not self.interrupt_event.is_set():

            # Run processing with the correct frequency
            next_call = next_call+self.graph_update_period
            sleep_time = next_call - time.time()
            if sleep_time > 0.0:
                time.sleep(sleep_time)
            else:
                print('Missed control loop frequency')

            # Combine the rows into a single vector
            feature_activation = np.sum(self.sensor_attention_data, axis=0)
            feature_activation -= np.min(feature_activation)

            if not self.plot_attention_matrix:
                # Make the weight vector and correct its size
                upsample_factor = np.ceil(float(self.n_scans) / feature_activation.shape[0])
                attention_weights = gaussian_filter(feature_activation.repeat(upsample_factor,
                    axis=0), sigma=5)

                cut_elements = (attention_weights.shape[0] - self.n_scans) / 2
                attention_weights = attention_weights[cut_elements:-cut_elements]

                # Update the map data
                data = dict()
                data['start_angle'] = self.laser_plot.data_source.data['start_angle']
                data['end_angle'] = self.laser_plot.data_source.data['end_angle']
                data['outer_radius'] = np.minimum(self.laser_data, 10.0)
                data['attention'] = attention_weights
                self.laser_data_source.data = data

                self.target_plot.data_source.data = dict(x=[0.0, self.target_pose[1]],y=[0.0, self.target_pose[0]])
                self.heading_plot.data_source.data['right'] = [self.target_pose[2]]

            else:
                # Update the matrix data
                self.sensor_attention_source.data['attention'] = self.sensor_attention_data.flatten()

                self.sensor_attention_features.data_source.data['top'] = feature_activation
                self.sensor_attention_filters.data_source.data['right'] = np.sum(self.sensor_attention_data, axis=1)

    def set_laser_data(self, data):
        step = len(data)//self.n_scans
        self.laser_data = data[::step]

    def set_target_pose(self, pose):
        self.target_pose = pose

    def set_sensor_attention(self, attention):
        if attention.shape[0] != self.n_filter or attention.shape[1] != self.n_feature:
            print('Adjust the plotting size: {}*{}'.format(attention.shape[0], attention.shape[1]))

        if self.plot_attention_matrix:
            self.sensor_attention_data = np.fliplr(attention)
        else:
            self.sensor_attention_data = attention

    def create_plot(self):

        if not self.plot_attention_matrix:
            self.create_map_plot()
        else:
            self.crete_matrix_plot()

    def create_map_plot(self):
        """Create a plot which includes the laser scan with weights and the target pose"""

        self.mapper = LinearColorMapper(palette=brewer['YlOrRd'][9], low=self.attention_scale_max, high=0.0)
        
        # Plot laser data and target into a map
        two_seventy = np.pi*135.0/180.0
        theta = np.arange(-two_seventy, two_seventy+two_seventy/self.n_scans, 2*two_seventy/self.n_scans)
        theta += np.pi/2.0
        self.map = bkp.figure(x_range=(-10,10), y_range=(-10,10))

        self.laser_data_source = ColumnDataSource(data=dict(outer_radius=self.laser_data,
            attention=np.zeros([self.n_scans])))

        self.laser_plot = self.map.annular_wedge(inner_radius=0.0, outer_radius='outer_radius',
                start_angle=theta[:-1], end_angle=theta[1:],
                fill_color={'field': 'attention', 'transform': self.mapper},
                line_color=None, source=self.laser_data_source)

        self.target_plot = self.map.line([0.0,self.target_pose[0]], [0.0, self.target_pose[1]],
                line_width = 3)
        self.plot_bot(self.map, 0.0, 0.0, 0.0, 'grey')

        # Plot bar for the difference to the goal heading
        self.heading = bkp.figure(x_range=(-np.pi, np.pi), height=100, tools=[])
        self.heading.yaxis.visible = False
        self.heading_plot = self.heading.hbar(y=[1.0], height=0.5, left=[0.0],
                right=[0.0], color='firebrick')
    
        self.plots = column(self.map)

    def crete_matrix_plot(self):
        """Create a plot which includes a matrx of attention and its combination as bar charts"""

        self.mapper = LinearColorMapper(palette=brewer['YlOrRd'][9], low=self.attention_scale_max, high=0.0)

        # Plot the attention matrix
        x_index_matrix = np.arange(self.n_feature).reshape([self.n_feature,1]).repeat(self.n_filter, axis=1)
        y_index_matrix = np.arange(self.n_filter).reshape([self.n_filter,1]).repeat(self.n_feature, axis=1)
        x_range = x_index_matrix.transpose().flatten()
        y_range = y_index_matrix.flatten()
        self.sensor_attention_source = ColumnDataSource(data=dict(x=x_range, y=y_range,
            attention=self.sensor_attention_data.flatten()))

        TOOLS = "hover,save"

        self.sensor_attention = bkp.figure(x_range=(-1,self.n_feature), y_range=(-1,self.n_filter),
                tools=TOOLS)
        self.sensor_attention_plot = self.sensor_attention.rect(x='x', y='y', width=1,
                height=1, fill_color={'field': 'attention', 'transform': self.mapper},
                line_color=None, source=self.sensor_attention_source)

        self.sensor_attention.select_one(HoverTool).tooltips = [
                    ('x - y', '@x - @y'),
                    ('attention', '@attention')
                ]

        # Plot combined values of the filters
        self.filter_bars = bkp.figure(y_range=(-1,self.n_filter),x_range=(0.0,self.attention_scale_max), width=250)
        self.sensor_attention_filters = self.filter_bars.hbar(y=np.arange(0,self.n_filter),
                height=1,left=0,right=np.sum(self.sensor_attention_data, axis=1), color='firebrick')

        # Plot combined values of the features
        self.feature_bars = bkp.figure(x_range=(-1,self.n_feature), y_range=(0.0,self.attention_scale_max), height=250)
        self.sensor_attention_features = self.feature_bars.vbar(x=np.arange(0,self.n_feature),
                width=1,bottom=0,top=np.sum(self.sensor_attention_data, axis=0), color='firebrick')

        self.plots = column(row(self.sensor_attention, self.filter_bars), self.feature_bars)

    def plot_bot(self, plot, x,y,h, color):
        """Plot a circle and a direction marker onto the given plot"""
        h += np.pi/2
        
        robot = plot.wedge(x,y, radius=0.15, start_angle=-np.pi,
                end_angle=np.pi, color=color)
        marker = plot.wedge(x,y, radius=0.15, start_angle=h-0.25,
                end_angle=h+0.25, color='black')

        return (robot, marker)

        

