#!/usr/bin/env python3

import numpy as np


def radial_subsampling(data:np.ndarray, factor:float)->np.ndarray:
    """Performs a radial subsampling of the data array by keeping
    factor*total_number_of_angles angles and filling the rest with zeros



    Parameters
    ----------
    data : np.ndarray
        Array to be subsampled
    factor : float
        fraction of angles to be kept

    Returns
    -------
    np.ndarray
        Zero filled susampling of the data array

    """
    tab = np.zeros(data.shape, dtype=data.dtype)
    circumference = 2 * np.sum(data.shape)
    number_angles = int((circumference - 4)/factor)
    thetas = np.linspace(0, 2*np.pi, number_angles)
    yc, xc = data.shape[0]//2, data.shape[1]//2
    for theta in thetas:
        x, y = compute_stop_point(xc, yc, data.shape[1] - 1, data.shape[0] - 1, theta)
        dx = abs(x - xc)
        dy = abs(y - yc)

        if dx > dy:
            plot_pixel(xc, yc, x, y, dx, dy, 0, data, tab)
        else:
            plot_pixel(yc, xc, y, x, dy, dx, 1, data, tab)
    return tab


def put_data(x: int, y: int, data:np.ndarray, tab: np.ndarray)->None:
    """put data from the data array in the tab array



    Parameters
    ----------
    x : int
        x coordinate of the data
    y : int
        y coordinate of the data
    data : np.ndarray
        array to take data from
    tab : np.ndarray
        array to put data in
    """
    tab[y, x] = data[y, x]


def plot_pixel(x1: int, y1: int, x2: int, y2: int, dx: int, dy: int,
               decide: int, data: np.ndarray, tab: np.ndarray
               )->np.ndarray:
    """draw a line between 2 points in an image with data taken from the
    data array and put in the tab array using Bresenham's line
    algorithm: https://en.wikipedia.org/wiki/Bresenham%27s_line_algorithm
    Inspired by the following blog post:
    https://www.geeksforgeeks.org/bresenhams-line-generation-algorithm/


    Parameters
    ----------
    x1 : int
        x coordinate of the starting point of the line
    y1 : int
        y coordinate of the starting point of the line
    x2 : int
        x coordinate of the ending point of the line
    y2 : int
        y coordinate of the ending point of the line
    dx : int
        length of the line on the x axis
    dy : int
        length of the line on the y axis
    decide : int
        0 or 1. Used to distinguished the different cases of the
        algorithm
    data : np.ndarray
        The numpy array to take data from
    tab : np.ndarray
        The numpy array to put data in

    Returns
    -------
    np.ndarray


    """
    pk = 2 * dy - dx
    put_data(x1, y1, data, tab)
    for _ in range(dx+1):
        x1 = (x1 + 1) if x1<x2 else (x1 - 1)
        if pk < 0:
            if decide == 0:
                put_data(x1, y1, data, tab)
                pk = pk + 2 * dy
            else:
                put_data(y1, x1, data, tab)
                pk = pk + 2 * dy
        else:
            y1 = (y1 + 1) if y1<y2 else (y1 - 1)
            if (decide == 0):
                put_data(x1, y1, data, tab)
            else:
                put_data(y1, x1, data, tab)
            pk = pk + 2 * dy - 2 * dx
    return tab


def compute_stop_point(xc: int, yc: int, x_max: int, y_max: int,
                       theta: float)->tuple[int,int]:
    """Computes the intersection of the line coming from the
    point given by the coordinates (xc, yc) with an angle theta
    with the lines of equation x=0, y=0, x=x_max or y=y_max


    Parameters
    ----------
    xc : int
        x coordinate of the point chosen as center
    yc : int
        y coordinate of the point chosen as center
    x_max : int
        Maximum x coordinate
    y_max : int
        Maximum y coordinate
    theta : float
        Angle with which the line is coming from the center point

    Returns
    -------
    tuple[int,int]
        coordinates of the intersection

    """
    if np.pi/4 <= theta <= 3 * (np.pi/4):
        y = y_max
        x = xc + np.tan(np.pi/2 - theta) * (y - yc)

    elif 5 * np.pi/4 <= theta <= 7 * np.pi/4:
        y = 0
        x = xc + np.tan(np.pi/2 - theta) * (y - yc)

    elif 3 * np.pi/4 < theta < 5 * np.pi/4:
        x = 0
        y = yc + np.tan(theta) * (x -xc)

    else:
        x = x_max
        y = yc + np.tan(theta) * (x -xc)

    return int(x), int(y)
