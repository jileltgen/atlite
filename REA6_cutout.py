#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 24 15:43:35 2020

@author: jileltgen
"""

import atlite

import logging
logging.basicConfig(level=logging.INFO)

cutout = atlite.Cutout(path="western-europe-2018.nc",
                       rea_dir=r"F:\weather_data\REA",
                       module="REA6",
                       x=slice(-44.67010498046875, 65.05841064453124),
                       y=slice(21.95256, 72.36798119565216),
                       dx = 0.12939683446344338,
                       dy =  0.06118376358695652,
                       time="2018",
                       chunks={'time': 100}
                       )

cutout.prepare()

#print(cutout.available_features.to_frame())

#print(cutout.data)

