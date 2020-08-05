#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 24 15:43:35 2020

@author: jileltgen
"""

import atlite

import logging
logging.basicConfig(level=logging.INFO)

cutout = atlite.Cutout(path="western-europe-1995.nc",
                       module="REA6",
                       x=slice(-19.0, 45.0),
                       y=slice(33.0, 75.0),
                       dx = 0.055,
                       dy = 0.055,
                       time="1995",
                       chunks={'time': 100}
                       )

cutout.prepare()

#print(cutout.available_features.to_frame())

#print(cutout.data)

