---
layout: post
title: Creating a map and KDE plot of points and polygons with Python
date: 2020-01-22 16:46:20 +0300
description: This tutorial teaches you how to plot map data on a background map of OpenStreetMap using Python. 
img:  /post1/teaser.jpg
tags: [Python, Maps, Matplotlib, Visualization, Pandas, Geodata] 
---

## ! add in updated code from colab and the other notebook !


This tutorial teaches you how to plot map data on a background map of OpenStreetMap using Python. The [tutorial is in form of a Jupyter Notebook](code.ipynb), therefore you either install Jupyter Lab or you can also copy the code into any other editor. The results should look like the following images:



<div class="row">
  <div class="column">
        <img src="{{site.baseurl}}/assets/img/post1/map1.jpg" height="200" width="200"/>
        <p>This is image 1</p>
    </div>
  <div class="column">
        <img class="middle-img" src="{{site.baseurl}}/assets/img/post1/map2.jpg" height="200" width="200"/>
        <p>This is image 2</p>
    </div>
 <div class="column">
         <img src="{{site.baseurl}}/assets/img/post1/map3.jpg" height="200" width="200"/>
        <p>This is image 3</p>
    </div>
</div>


## Installation

This tutorial requires the installation of multiple packages, a few of them are not installable for windows under pip. Therefore, the packages

{% highlight bash %}
FionaGDAL
Rtree
Shapely
{% endhighlight %}

can be installed as wheels with this code:

{% highlight bash %}
pip install .\package_wheels_windows\Fiona-1.8.9-cp37-cp37m-win_amd64.whl
pip install .\package_wheels_windows\GDAL-3.0.1-cp37-cp37m-win_amd64.whl
pip install .\package_wheels_windows\Rtree-0.8.3-cp37-cp37m-win_amd64.whl
pip install .\package_wheels_windows\Shapely-1.6.4.post2-cp37-cp37m-win_amd64.whl
{% endhighlight %}
After that, the rest of the packages should be easily installable by using the provided requirements.txt file:

{% highlight bash %}
pip install -r requirements.txt
{% endhighlight %}

## Importing packages

{% highlight python %}
import numpy as np 
import pandas as pd
import geopandas as gpd 
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.path import Path
from matplotlib.textpath import TextToPath
import tilemapbase
import warnings
import matplotlib.cbook
warnings.filterwarnings("ignore",category=matplotlib.cbook.mplDeprecation)

import seaborn as sns
import shapely.speedups
shapely.speedups.enable()
{% endhighlight %}

Download the following data and save extract it into the ./data folder of this project:

- http://download.geofabrik.de/europe/netherlands/noord-holland-latest-free.shp.zip
- https://maps.amsterdam.nl/open_geodata/geojson.php?KAARTLAAG=GEBIED_STADSDELEN&THEMA=gebiedsindeling

## Loading and preprocessing the data

Load the POI data:
{% highlight python %}
points = gpd.read_file("./data/gis_osm_pois_free_1.shp")
points = points.to_crs({"init": "EPSG:3857"})
{% endhighlight %}

filter the data for restaurants (or any other POI category)
{% highlight python %}
points = points[points["fclass"] == "restaurant"]
{% endhighlight %}
take a quick look into the data
{% highlight python %}
print(points)
{% endhighlight %}

| osm_id | code       | fclass | name       | geometry           |                                |
|--------|------------|--------|------------|--------------------|--------------------------------|
| 92     | 30839687   | 2301   | restaurant | de Eethoek         | POINT (521775.259 6910003.671) |
| 144    | 34043796   | 2301   | restaurant | Sizzling Wok       | POINT (550760.350 6853264.008) |
| ...    | ...        | ...    | ...        | ...                | ...                            |
| 37100  | 7126562155 | 2301   | restaurant | Duinberk           | POINT (522497.422 6928134.759) |
| 37111  | 7137254485 | 2301   | restaurant | Vleesch noch Visch | POINT (542280.944 6869060.363) |

load the shapefile for the city of amsterdam
{% highlight python %}
city = gpd.read_file("./data/geojson.json")
city = city.to_crs({"init": "EPSG:3857"})
{% endhighlight %}
take a quick look into this data as well
{% highlight python %}
print(city)
{% endhighlight %}
| Stadsdeel_code | Stadsdeel | Opp_m2     | geometry |                                                   |
|----------------|-----------|------------|----------|---------------------------------------------------|
| 0              | A         | Centrum    | 8043500  | POLYGON ((549136.599 6867376.523, 549133.148 6... |
| 1              | B         | Westpoort  | 28991600 | POLYGON ((543892.115 6872660.218, 543540.457 6... |
| 2              | E         | West       | 10629900 | POLYGON ((544918.815 6870710.847, 544873.285 6... |
| 3              | F         | Nieuw-West | 38015500 | POLYGON ((539955.524 6866252.019, 539951.183 6... |
| 4              | K         | Zuid       | 17274000 | POLYGON ((547134.629 6862225.476, 547129.731 6... |
| 5              | M         | Oost       | 30594900 | POLYGON ((560946.039 6864490.649, 560918.543 6... |
| 6              | N         | Noord      | 63828800 | POLYGON ((565410.507 6870704.099, 564865.041 6... |
| 7              | T         | Zuidoost   | 22113700 | POLYGON ((558996.500 6854997.221, 558987.372 6... |

perform a spatial join between the points and polygons and filter out any points that did not match with the polygons
{% highlight python %}
points = gpd.sjoin(points, city, how="left")
points = points.dropna(subset=["index_right"])
{% endhighlight %}
plot both data sets to see if the spatial join was performed correctly
{% highlight python %}
# edit the figure size however you need to
plt.figure(num=None, figsize=(10,10), dpi=80, facecolor='w', edgecolor='k')
# create plot and axes
fig = plt.plot()
ax1 = plt.axes()
# these values can be changed as needed, the markers are LaTeX symbols
city.plot(ax=ax1, alpha=0.1, edgecolor="black", facecolor="white")
points.plot(ax=ax1, alpha = 0.1, color="red", marker='$\\bigtriangledown$',)
ax1.figure.savefig('./data/plot1.png', bbox_inches='tight')
{% endhighlight %}

## Add a background map to the plot

<img align="right" src="{{site.baseurl}}/assets/img/post1/map3.jpg">
to get a nice background map, we need to find out all the boundaries of our data and save that for later plotting
{% highlight python %}
bounding_box = [points["geometry"].x.min(), points["geometry"].x.max(), points["geometry"].y.min(), points["geometry"].y.max()]
{% endhighlight %}
load the background map using `tilemapbase`
{% highlight python %}
tilemapbase.start_logging()
tilemapbase.init(create=True)
extent = tilemapbase.extent_from_frame(city, buffer = 25)
{% endhighlight %}
{% highlight python %}
fig, ax = plt.subplots(figsize=(10,10))

plotter = tilemapbase.Plotter(extent, tilemapbase.tiles.build_OSM(), width=1000)
plotter.plot(ax)
ax.set_xlim(bounding_box[0]+2000, bounding_box[1])
ax.set_ylim(bounding_box[2]+2000, bounding_box[3])
city.plot(ax=ax, alpha=0.3, edgecolor="black", facecolor="white")
points.plot(ax=ax, alpha = 0.4, color="red", marker='$\\bigtriangledown$',)
ax.figure.savefig('./data/plot1.png', bbox_inches='tight')
{% endhighlight %}

## Show a KDE plot of the spatial distribution

To get an impression on the spatial distribution, a KDE plot might help. For this we use the `kdeplot`-function from seaborn.

{% highlight python %}
fig, ax = plt.subplots(figsize=(10,10))

plotter = tilemapbase.Plotter(extent, tilemapbase.tiles.build_OSM(), width=1000)
plotter.plot(ax)
ax.set_xlim(bounding_box[0]+2000, bounding_box[1])
ax.set_ylim(bounding_box[2]+2000, bounding_box[3])
city.plot(ax=ax, alpha=0.3, edgecolor="black", facecolor="white")
sns.kdeplot(points["geometry"].x, points["geometry"].y, shade=True, alpha=0.5, cmap="viridis", shade_lowest=False)
ax.figure.savefig('./data/plot2.png', bbox_inches='tight')
{% endhighlight %}

## Color the markers with the KDE information

instead of drawing the KDE as a single shape, we can also color our points according to the density. For this we calculate the gaussian KDE separately and use the result as z-values for our plot. The markers can be changed to ones liking, for this case I settled with simple points.
{% highlight python %}
xy = np.vstack([points["geometry"].x,points["geometry"].y])
z = gaussian_kde(xy)(xy)
{% endhighlight %}
{% highlight python %}
fig, ax = plt.subplots(figsize=(10,10))

plotter = tilemapbase.Plotter(extent, tilemapbase.tiles.build_OSM(), width=1000)
plotter.plot(ax)
ax.set_xlim(bounding_box[0]+2000, bounding_box[1])
ax.set_ylim(bounding_box[2]+2000, bounding_box[3])
city.plot(ax=ax, alpha=0.3, edgecolor="black", facecolor="white")
ax.scatter(points["geometry"].x, points["geometry"].y, c=z, s=20, zorder=2, edgecolor='',  alpha=0.7)
ax.figure.savefig('./data/plot3.png', bbox_inches='tight')
{% endhighlight %}

## Use more specific symbols as map markers

If you want to change the markers in the map to more sophisticated ones, you could also use Font Awesome. Download the font from here and save it to `/resources/`. Edit the `symbols` dict to add symbols that might fit your subject, a cheat sheet for the unicode characters can be found on [the fontawesome website](https://fontawesome.com/cheatsheet). Just add a `\u` to any of the unicode characters.
{% highlight python %}
fp = FontProperties(fname=r"./resources/Font Awesome 5 Free-Solid-900.otf") 

def get_marker(symbol):
    v, codes = TextToPath().get_text_path(fp, symbol)
    v = np.array(v)
    mean = np.mean([np.max(v,axis=0), np.min(v, axis=0)], axis=0)
    return Path(v-mean, codes, closed=False)

symbols = dict(map = "\uf041", map_alt = "\uf3c5")
{% endhighlight %}

{% highlight python %}
fig, ax = plt.subplots(figsize=(10,10))

plotter = tilemapbase.Plotter(extent, tilemapbase.tiles.build_OSM(), width=1000)
plotter.plot(ax)
ax.set_xlim(bounding_box[0]+2000, bounding_box[1])
ax.set_ylim(bounding_box[2]+2000, bounding_box[3])
city.plot(ax=ax, alpha=0.3, edgecolor="black", facecolor="white")
ax.scatter(points["geometry"].x, points["geometry"].y, c="red", s=35, zorder=2, edgecolor='',  alpha=0.5, marker=get_marker(symbols["map"]))
ax.figure.savefig('./data/plot4.png', bbox_inches='tight')
{% endhighlight %}
{% highlight python %}
fig, ax = plt.subplots(figsize=(10,10))

plotter = tilemapbase.Plotter(extent, tilemapbase.tiles.build_OSM(), width=1000)
plotter.plot(ax)
ax.set_xlim(bounding_box[0]+2000, bounding_box[1])
ax.set_ylim(bounding_box[2]+2000, bounding_box[3])
city.plot(ax=ax, alpha=0.3, edgecolor="black", facecolor="white")
ax.scatter(points["geometry"].x, points["geometry"].y, c=z, s=35, zorder=2, edgecolor='',  alpha=0.5, marker=get_marker(symbols["map"]))
ax.figure.savefig('./data/plot5.png', bbox_inches='tight')
{% endhighlight %}