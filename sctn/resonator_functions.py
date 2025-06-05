import numpy as np

from sctn.resonator import simple_resonator



###########################################

# Resonator functions for 0-10 Hz range

###########################################



def resonator_1_05():

    resonator = simple_resonator(

        freq0=1.05,

        clk_freq=15360,

        lf=4,

        thetas=[-11.912, -11.103, -9.652, -9.996],

        weights=[42.046, 18.636, 21.913, 19.553, 20.16],

    )

    resonator.log_out_spikes(-1)

    return resonator



def resonator_1_10():

    resonator = simple_resonator(

        freq0=1.10,

        clk_freq=15360,

        lf=4,

        thetas=[-12.327, -11.735, -9.997, -10.622],

        weights=[43.742, 19.585, 23.06, 20.334, 21.4],

    )

    resonator.log_out_spikes(-1)

    return resonator



def resonator_1_15():

    resonator = simple_resonator(

        freq0=1.15,

        clk_freq=153600,

        lf=4,

        thetas=[-13.016, -12.335, -10.737, -10.707],

        weights=[46.083, 20.491, 24.346, 21.682, 21.655],

    )

    resonator.log_out_spikes(-1)

    return resonator



def resonator_1_28():

    resonator = simple_resonator(

        freq0=1.28,

        clk_freq=15360,

        lf=4,

        thetas=[-14.609, -13.618, -11.787, -12.061],

        weights=[51.511, 22.744, 26.914, 23.81, 24.306],

    )

    resonator.log_out_spikes(-1)

    return resonator



def resonator_1_30():

    resonator = simple_resonator(

        freq0=1.30,

        clk_freq=15360,

        lf=4,

        thetas=[-38.705, -14.082, -11.844, -12.294],

        weights=[97.872, 21.307, 27.448, 24.055, 24.879],

    )

    resonator.log_out_spikes(-1)

    return resonator



def resonator_1_59():

    resonator = simple_resonator(

        freq0=1.59,

        clk_freq=15360,

        lf=4,

        thetas=[-18.833, -17.091, -14.972, -15.48],

        weights=[63.287, 26.537, 33.677, 30.133, 31.076],

    )

    resonator.log_out_spikes(-1)

    return resonator



def resonator_1_60():

    resonator = simple_resonator(

        freq0=1.60,

        clk_freq=15360,

        lf=4,

        thetas=[-19.184, -17.323, -15.152, -15.803],

        weights=[63.808, 26.451, 34.141, 30.497, 31.678],

    )

    resonator.log_out_spikes(-1)

    return resonator



def resonator_1_66():

    resonator = simple_resonator(

        freq0=1.66,

        clk_freq=15360,

        lf=4,

        thetas=[-20.0, -17.797, -15.956, -16.104],

        weights=[66.282, 27.3, 35.106, 32.001, 32.329],

    )

    resonator.log_out_spikes(-1)

    return resonator



def resonator_1_75():

    resonator = simple_resonator(

        freq0=1.75,

        clk_freq=15360,

        lf=4,

        thetas=[-24.442, -17.659, -17.053, -16.969],

        weights=[78.467, 29.833, 35.414, 34.069, 33.882],

    )

    resonator.log_out_spikes(-1)

    return resonator



def resonator_1_90():

    resonator = simple_resonator(

        freq0=1.90,

        clk_freq=15360,

        lf=4,

        thetas=[-22.972, -20.365, -18.31, -18.796],

        weights=[75.971, 31.307, 40.053, 36.719, 37.694],

    )

    resonator.log_out_spikes(-1)

    return resonator



def resonator_1_95():

    resonator = simple_resonator(

        freq0=1.95,

        clk_freq=15360,

        lf=4,

        thetas=[-23.292, -20.967, -18.739, -19.253],

        weights=[77.357, 32.09, 41.255, 37.596, 38.596],

    )

    resonator.log_out_spikes(-1)

    return resonator



def resonator_2_21():

    resonator = simple_resonator(

        freq0=2.21,

        clk_freq=15360,

        lf=4,

        thetas=[-26.511, -23.624, -21.416, -21.755],

        weights=[87.574, 36.152, 46.334, 42.9, 43.672],

    )

    resonator.log_out_spikes(-1)

    return resonator



def resonator_2_50():

    resonator = simple_resonator(

        freq0=2.50,

        clk_freq=15360,

        lf=4,

        thetas=[-30.365, -26.786, -24.465, -24.437],

        weights=[99.455, 40.538, 52.741, 48.947, 49.019],

    )

    resonator.log_out_spikes(-1)

    return resonator



def resonator_2_68():

    resonator = simple_resonator(

        freq0=2.68,

        clk_freq=15360,

        lf=4,

        thetas=[-8.927, -25.704, -25.206, -26.618],

        weights=[65.351, 47.423, 51.518, 50.371, 53.112],

    )

    resonator.log_out_spikes(-1)

    return resonator



def resonator_2_79():

    resonator = simple_resonator(

        freq0=2.79,

        clk_freq=15360,

        lf=4,

        thetas=[-87.121, -27.251, -27.245, -28.08],

        weights=[223.098, 48.757, 55.048, 54.062, 55.662],

    )

    resonator.log_out_spikes(-1)

    return resonator



def resonator_2_88():

    resonator = simple_resonator(

        freq0=2.88,

        clk_freq=15360,

        lf=4,

        thetas=[-38.576, -29.943, -28.106, -27.831],

        weights=[123.81, 48.212, 59.177, 56.132, 55.85],

    )

    resonator.log_out_spikes(-1)

    return resonator



def resonator_3_05():

    resonator = simple_resonator(

        freq0=3.05,

        clk_freq=15360,

        lf=4,

        thetas=[-37.124, -32.549, -29.814, -29.008],

        weights=[123.204, 50.851, 64.454, 59.25, 58.383],

    )

    resonator.log_out_spikes(-1)

    return resonator



def resonator_3_39():

    resonator = simple_resonator(

        freq0=3.39,

        clk_freq=15360,

        lf=4,

        thetas=[-45.112, -36.415, -32.509, -32.316],

        weights=[144.962, 56.675, 72.227, 64.768, 64.959],

    )

    resonator.log_out_spikes(-1)

    return resonator



def resonator_3_47():

    resonator = simple_resonator(

        freq0=3.47,

        clk_freq=15360,

        lf=4,

        thetas=[-44.809, -37.682, -34.249, -33.187],

        weights=[142.565, 55.948, 73.91, 68.354, 67.02],

    )

    resonator.log_out_spikes(-1)

    return resonator



def resonator_3_72():

    resonator = simple_resonator(

        freq0=3.72,

        clk_freq=15360,

        lf=4,

        thetas=[-48.663, -40.06, -35.864, -35.336],

        weights=[157.321, 62.183, 79.307, 71.374, 71.063],

    )

    resonator.log_out_spikes(-1)

    return resonator



def resonator_4_02():

    resonator = simple_resonator(

        freq0=4.02,

        clk_freq=15360,

        lf=4,

        thetas=[-52.983, -43.606, -39.073, -37.241],

        weights=[170.532, 67.256, 86.596, 77.271, 75.25],

    )

    resonator.log_out_spikes(-1)

    return resonator



def resonator_4_12():

    resonator = simple_resonator(

        freq0=4.12,

        clk_freq=15360,

        lf=4,

        thetas=[-54.501, -44.59, -39.86, -38.707],

        weights=[175.469, 69.046, 88.335, 79.108, 78.08],

    )

    resonator.log_out_spikes(-1)

    return resonator



def resonator_4_36():

    resonator = simple_resonator(

        freq0=4.36,

        clk_freq=15360,

        lf=4,

        thetas=[-58.654, -47.541, -42.627, -40.103],

        weights=[187.055, 72.849, 94.515, 84.064, 81.135],

    )

    resonator.log_out_spikes(-1)

    return resonator



def resonator_4_62():

    resonator = simple_resonator(

        freq0=4.62,

        clk_freq=15360,

        lf=4,

        thetas=[-61.279, -49.554, -45.289, -44.119],

        weights=[194.982, 75.689, 97.964, 90.261, 89.009],

    )

    resonator.log_out_spikes(-1)

    return resonator



def resonator_4_77():

    resonator = simple_resonator(

        freq0=4.77,

        clk_freq=15360,

        lf=4,

        thetas=[-62.781, -51.17, -46.907, -45.359],

        weights=[200.524, 78.397, 101.374, 92.831, 91.37],

    )

    resonator.log_out_spikes(-1)

    return resonator



def resonator_5_09():

    resonator = simple_resonator(

        freq0=5.09,

        clk_freq=15360,

        lf=4,

        thetas=[-69.695, -55.164, -50.732, -47.095],

        weights=[219.153, 83.76, 109.522, 99.543, 95.232],

    )

    resonator.log_out_spikes(-1)

    return resonator



def resonator_5_26():

    resonator = simple_resonator(

        freq0=5.26,

        clk_freq=15360,

        lf=4,

        thetas=[-78.345, -57.279, -52.931, -48.217],

        weights=[239.144, 86.311, 114.419, 103.404, 97.135],

    )

    resonator.log_out_spikes(-1)

    return resonator



def resonator_5_45():

    resonator = simple_resonator(

        freq0=5.45,

        clk_freq=15360,

        lf=4,

        thetas=[-81.139, -59.604, -54.623, -49.983],

        weights=[247.52, 89.124, 119.23, 106.752, 100.593],

    )

    resonator.log_out_spikes(-1)

    return resonator



def resonator_5_87():

    resonator = simple_resonator(

        freq0=5.87,

        clk_freq=15360,

        lf=4,

        thetas=[-88.619, -64.356, -59.071, -53.569],

        weights=[268.708, 95.611, 128.823, 115.302, 107.954],

    )

    resonator.log_out_spikes(-1)

    return resonator



def resonator_6_36():

    resonator = simple_resonator(

        freq0=6.36,

        clk_freq=15360,

        lf=4,

        thetas=[-98.054, -70.205, -64.087, -57.686],

        weights=[294.356, 103.068, 140.527, 124.915, 116.543],

    )

    resonator.log_out_spikes(-1)

    return resonator



def resonator_6_94():

    resonator = simple_resonator(

        freq0=6.94,

        clk_freq=15360,

        lf=4,

        thetas=[-113.694, -77.332, -69.207, -62.432],

        weights=[334.262, 112.405, 154.573, 135.118, 126.891],

    )

    resonator.log_out_spikes(-1)

    return resonator



def resonator_7_63():

    resonator = simple_resonator(

        freq0=7.63,

        clk_freq=15360,

        lf=4,

        thetas=[-127.835, -85.872, -76.422, -67.925],

        weights=[371.729, 122.996, 171.202, 148.778, 139.294],

    )

    resonator.log_out_spikes(-1)

    return resonator



def resonator_8_98():

    resonator = simple_resonator(

        freq0=8.98,

        clk_freq=15360,

        lf=4,

        thetas=[-106.516, -93.167, -90.835, -81.611],

        weights=[359.026, 152.001, 182.454, 179.91, 167.481],

    )

    resonator.log_out_spikes(-1)

    return resonator



def resonator_9_54():

    resonator = simple_resonator(

        freq0=9.54,

        clk_freq=15360,

        lf=4,

        thetas=[-117.549, -100.74, -96.393, -85.843],

        weights=[387.993, 160.185, 197.341, 190.822, 177.199],

    )

    resonator.log_out_spikes(-1)

    return resonator



###########################################

# Resonator functions for 10-100 Hz range

###########################################



def resonator_10_5():

    resonator = simple_resonator(

        freq0=10.5,

        clk_freq=153600,

        lf=4,

        thetas=[-11.912, -11.103, -9.652, -9.996],

        weights=[42.046, 18.636, 21.913, 19.553, 20.16],

    )

    resonator.log_out_spikes(-1)

    return resonator



def resonator_11_0():

    resonator = simple_resonator(

        freq0=11.0,

        clk_freq=153600,

        lf=4,

        thetas=[-12.327, -11.735, -9.997, -10.622],

        weights=[43.742, 19.585, 23.06, 20.334, 21.4],

    )

    resonator.log_out_spikes(-1)

    return resonator



def resonator_11_5():

    resonator = simple_resonator(

        freq0=11.5,

        clk_freq=153600,

        lf=4,

        thetas=[-13.016, -12.335, -10.737, -10.707],

        weights=[46.083, 20.491, 24.346, 21.682, 21.655],

    )

    resonator.log_out_spikes(-1)

    return resonator



def resonator_12_8():

    resonator = simple_resonator(

        freq0=12.8,

        clk_freq=153600,

        lf=4,

        thetas=[-14.609, -13.618, -11.787, -12.061],

        weights=[51.511, 22.744, 26.914, 23.81, 24.306],

    )

    resonator.log_out_spikes(-1)

    return resonator



def resonator_13_0():

    resonator = simple_resonator(

        freq0=13.0,

        clk_freq=153600,

        lf=4,

        thetas=[-38.705, -14.082, -11.844, -12.294],

        weights=[97.872, 21.307, 27.448, 24.055, 24.879],

    )

    resonator.log_out_spikes(-1)

    return resonator



def resonator_15_9():

    resonator = simple_resonator(

        freq0=15.9,

        clk_freq=153600,

        lf=4,

        thetas=[-18.833, -17.091, -14.972, -15.48],

        weights=[63.287, 26.537, 33.677, 30.133, 31.076],

    )

    resonator.log_out_spikes(-1)

    return resonator



def resonator_16_0():

    resonator = simple_resonator(

        freq0=16.0,

        clk_freq=153600,

        lf=4,

        thetas=[-19.184, -17.323, -15.152, -15.803],

        weights=[63.808, 26.451, 34.141, 30.497, 31.678],

    )

    resonator.log_out_spikes(-1)

    return resonator



def resonator_16_6():

    resonator = simple_resonator(

        freq0=16.6,

        clk_freq=153600,

        lf=4,

        thetas=[-20.0, -17.797, -15.956, -16.104],

        weights=[66.282, 27.3, 35.106, 32.001, 32.329],

    )

    resonator.log_out_spikes(-1)

    return resonator



def resonator_17_5():

    resonator = simple_resonator(

        freq0=17.5,

        clk_freq=153600,

        lf=4,

        thetas=[-24.442, -17.659, -17.053, -16.969],

        weights=[78.467, 29.833, 35.414, 34.069, 33.882],

    )

    resonator.log_out_spikes(-1)

    return resonator



def resonator_19_0():

    resonator = simple_resonator(

        freq0=19.0,

        clk_freq=153600,

        lf=4,

        thetas=[-22.972, -20.365, -18.31, -18.796],

        weights=[75.971, 31.307, 40.053, 36.719, 37.694],

    )

    resonator.log_out_spikes(-1)

    return resonator



def resonator_19_5():

    resonator = simple_resonator(

        freq0=19.5,

        clk_freq=153600,

        lf=4,

        thetas=[-23.292, -20.967, -18.739, -19.253],

        weights=[77.357, 32.09, 41.255, 37.596, 38.596],

    )

    resonator.log_out_spikes(-1)

    return resonator



def resonator_22_1():

    resonator = simple_resonator(

        freq0=22.1,

        clk_freq=153600,

        lf=4,

        thetas=[-26.511, -23.624, -21.416, -21.755],

        weights=[87.574, 36.152, 46.334, 42.9, 43.672],

    )

    resonator.log_out_spikes(-1)

    return resonator



def resonator_25_0():

    resonator = simple_resonator(

        freq0=25.0,

        clk_freq=153600,

        lf=4,

        thetas=[-30.365, -26.786, -24.465, -24.437],

        weights=[99.455, 40.538, 52.741, 48.947, 49.019],

    )

    resonator.log_out_spikes(-1)

    return resonator



def resonator_26_8():

    resonator = simple_resonator(

        freq0=26.8,

        clk_freq=153600,

        lf=4,

        thetas=[-8.927, -25.704, -25.206, -26.618],

        weights=[65.351, 47.423, 51.518, 50.371, 53.112],

    )

    resonator.log_out_spikes(-1)

    return resonator



def resonator_27_9():

    resonator = simple_resonator(

        freq0=27.9,

        clk_freq=153600,

        lf=4,

        thetas=[-87.121, -27.251, -27.245, -28.08],

        weights=[223.098, 48.757, 55.048, 54.062, 55.662],

    )

    resonator.log_out_spikes(-1)

    return resonator



def resonator_28_8():

    resonator = simple_resonator(

        freq0=28.8,

        clk_freq=153600,

        lf=4,

        thetas=[-38.576, -29.943, -28.106, -27.831],

        weights=[123.81, 48.212, 59.177, 56.132, 55.85],

    )

    resonator.log_out_spikes(-1)

    return resonator



def resonator_30_5():

    resonator = simple_resonator(

        freq0=30.5,

        clk_freq=153600,

        lf=4,

        thetas=[-37.124, -32.549, -29.814, -29.008],

        weights=[123.204, 50.851, 64.454, 59.25, 58.383],

    )

    resonator.log_out_spikes(-1)

    return resonator



def resonator_33_9():

    resonator = simple_resonator(

        freq0=33.9,

        clk_freq=153600,

        lf=4,

        thetas=[-45.112, -36.415, -32.509, -32.316],

        weights=[144.962, 56.675, 72.227, 64.768, 64.959],

    )

    resonator.log_out_spikes(-1)

    return resonator



def resonator_34_7():

    resonator = simple_resonator(

        freq0=34.7,

        clk_freq=153600,

        lf=4,

        thetas=[-44.809, -37.682, -34.249, -33.187],

        weights=[142.565, 55.948, 73.91, 68.354, 67.02],

    )

    resonator.log_out_spikes(-1)

    return resonator



def resonator_37_2():

    resonator = simple_resonator(

        freq0=37.2,

        clk_freq=153600,

        lf=4,

        thetas=[-48.663, -40.06, -35.864, -35.336],

        weights=[157.321, 62.183, 79.307, 71.374, 71.063],

    )

    resonator.log_out_spikes(-1)

    return resonator



def resonator_40_2():

    resonator = simple_resonator(

        freq0=40.2,

        clk_freq=153600,

        lf=4,

        thetas=[-52.983, -43.606, -39.073, -37.241],

        weights=[170.532, 67.256, 86.596, 77.271, 75.25],

    )

    resonator.log_out_spikes(-1)

    return resonator



def resonator_41_2():

    resonator = simple_resonator(

        freq0=41.2,

        clk_freq=153600,

        lf=4,

        thetas=[-54.501, -44.59, -39.86, -38.707],

        weights=[175.469, 69.046, 88.335, 79.108, 78.08],

    )

    resonator.log_out_spikes(-1)

    return resonator



def resonator_43_6():

    resonator = simple_resonator(

        freq0=43.6,

        clk_freq=153600,

        lf=4,

        thetas=[-58.654, -47.541, -42.627, -40.103],

        weights=[187.055, 72.849, 94.515, 84.064, 81.135],

    )

    resonator.log_out_spikes(-1)

    return resonator



def resonator_46_2():

    resonator = simple_resonator(

        freq0=46.2,

        clk_freq=153600,

        lf=4,

        thetas=[-61.279, -49.554, -45.289, -44.119],

        weights=[194.982, 75.689, 97.964, 90.261, 89.009],

    )

    resonator.log_out_spikes(-1)

    return resonator



def resonator_47_7():

    resonator = simple_resonator(

        freq0=47.7,

        clk_freq=153600,

        lf=4,

        thetas=[-62.781, -51.17, -46.907, -45.359],

        weights=[200.524, 78.397, 101.374, 92.831, 91.37],

    )

    resonator.log_out_spikes(-1)

    return resonator



def resonator_50_9():

    resonator = simple_resonator(

        freq0=50.9,

        clk_freq=153600,

        lf=4,

        thetas=[-69.695, -55.164, -50.732, -47.095],

        weights=[219.153, 83.76, 109.522, 99.543, 95.232],

    )

    resonator.log_out_spikes(-1)

    return resonator



def resonator_52_6():

    resonator = simple_resonator(

        freq0=52.6,

        clk_freq=153600,

        lf=4,

        thetas=[-78.345, -57.279, -52.931, -48.217],

        weights=[239.144, 86.311, 114.419, 103.404, 97.135],

    )

    resonator.log_out_spikes(-1)

    return resonator



def resonator_54_5():

    resonator = simple_resonator(

        freq0=54.5,

        clk_freq=153600,

        lf=4,

        thetas=[-81.139, -59.604, -54.623, -49.983],

        weights=[247.52, 89.124, 119.23, 106.752, 100.593],

    )

    resonator.log_out_spikes(-1)

    return resonator



def resonator_58_7():

    resonator = simple_resonator(

        freq0=58.7,

        clk_freq=153600,

        lf=4,

        thetas=[-88.619, -64.356, -59.071, -53.569],

        weights=[268.708, 95.611, 128.823, 115.302, 107.954],

    )

    resonator.log_out_spikes(-1)

    return resonator



def resonator_63_6():

    resonator = simple_resonator(

        freq0=63.6,

        clk_freq=153600,

        lf=4,

        thetas=[-98.054, -70.205, -64.087, -57.686],

        weights=[294.356, 103.068, 140.527, 124.915, 116.543],

    )

    resonator.log_out_spikes(-1)

    return resonator



def resonator_69_4():

    resonator = simple_resonator(

        freq0=69.4,

        clk_freq=153600,

        lf=4,

        thetas=[-113.694, -77.332, -69.207, -62.432],

        weights=[334.262, 112.405, 154.573, 135.118, 126.891],

    )

    resonator.log_out_spikes(-1)

    return resonator



def resonator_76_3():

    resonator = simple_resonator(

        freq0=76.3,

        clk_freq=153600,

        lf=4,

        thetas=[-127.835, -85.872, -76.422, -67.925],

        weights=[371.729, 122.996, 171.202, 148.778, 139.294],

    )

    resonator.log_out_spikes(-1)

    return resonator



def resonator_89_8():

    resonator = simple_resonator(

        freq0=89.8,

        clk_freq=153600,

        lf=4,

        thetas=[-106.516, -93.167, -90.835, -81.611],

        weights=[359.026, 152.001, 182.454, 179.91, 167.481],

    )

    resonator.log_out_spikes(-1)

    return resonator



def resonator_95_4():

    resonator = simple_resonator(

        freq0=95.4,

        clk_freq=153600,

        lf=4,

        thetas=[-117.549, -100.74, -96.393, -85.843],

        weights=[387.993, 160.185, 197.341, 190.822, 177.199],

    )

    resonator.log_out_spikes(-1)

    return resonator



# Dictionary mapping all frequencies (0-100 Hz) to their resonator functions

RESONATOR_FUNCTIONS = {

    # 0-10 Hz range

    1.05: resonator_1_05,

    1.10: resonator_1_10,

    1.15: resonator_1_15,

    1.28: resonator_1_28,

    1.30: resonator_1_30,

    1.59: resonator_1_59,

    1.60: resonator_1_60,

    1.66: resonator_1_66,

    1.75: resonator_1_75,

    1.90: resonator_1_90,

    1.95: resonator_1_95,

    2.21: resonator_2_21,

    2.50: resonator_2_50,

    2.68: resonator_2_68,

    2.79: resonator_2_79,

    2.88: resonator_2_88,

    3.05: resonator_3_05,

    3.39: resonator_3_39,

    3.47: resonator_3_47,

    3.72: resonator_3_72,

    4.02: resonator_4_02,

    4.12: resonator_4_12,

    4.36: resonator_4_36,

    4.62: resonator_4_62,

    4.77: resonator_4_77,

    5.09: resonator_5_09,

    5.26: resonator_5_26,

    5.45: resonator_5_45,

    5.87: resonator_5_87,

    6.36: resonator_6_36,

    6.94: resonator_6_94,

    7.63: resonator_7_63,

    8.98: resonator_8_98,

    9.54: resonator_9_54,



    # 10-100 Hz range

    10.5: resonator_10_5,

    11.0: resonator_11_0,

    11.5: resonator_11_5,

    12.8: resonator_12_8,

    13.0: resonator_13_0,

    15.9: resonator_15_9,

    16.0: resonator_16_0,

    16.6: resonator_16_6,

    17.5: resonator_17_5,

    19.0: resonator_19_0,

    19.5: resonator_19_5,

    22.1: resonator_22_1,

    25.0: resonator_25_0,

    26.8: resonator_26_8,

    27.9: resonator_27_9,

    28.8: resonator_28_8,

    30.5: resonator_30_5,

    33.9: resonator_33_9,

    34.7: resonator_34_7,

    37.2: resonator_37_2,

    40.2: resonator_40_2,

    41.2: resonator_41_2,

    43.6: resonator_43_6,

    46.2: resonator_46_2,

    47.7: resonator_47_7,

    50.9: resonator_50_9,

    52.6: resonator_52_6,

    54.5: resonator_54_5,

    58.7: resonator_58_7,

    63.6: resonator_63_6,

    69.4: resonator_69_4,

    76.3: resonator_76_3,

    89.8: resonator_89_8,

    95.4: resonator_95_4

}



# Function to get the closest available resonator

def get_closest_resonator(target_freq):

    """

    Returns the resonator function with frequency closest to target_freq

    """

    available_freqs = list(RESONATOR_FUNCTIONS.keys())

    closest_freq = min(available_freqs, key=lambda x: abs(x - target_freq))

    return RESONATOR_FUNCTIONS[closest_freq], closest_freq



# Function to get resonator in a specific frequency band

def get_resonators_in_band(f_min, f_max):

    """

    Returns a list of (freq, resonator_func) for all resonators in the specified band

    """

    return [(f, RESONATOR_FUNCTIONS[f]) for f in sorted(RESONATOR_FUNCTIONS.keys())

            if f_min <= f <= f_max]
