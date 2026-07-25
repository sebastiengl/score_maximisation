import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import geopandas as gp
import pandas as p
import numpy as np
from shapely.geometry import Point, box
from matplotlib.patches import Polygon as MplPolygon, Patch
from matplotlib.collections import PatchCollection



########### CS1 ###########
# test_path = 'data/GLC24_PA_metadata_test.csv'
# train_path = 'data/GLC24_PA_metadata_train.csv'

# train_df = p.read_csv(train_path)
# test_df = p.read_csv(test_path)

# train_df = train_df.groupby('surveyId').first().reset_index()[['surveyId', 'lon', 'lat']]
# test_df = test_df.groupby('surveyId').first().reset_index()[['surveyId', 'lon', 'lat']]

# train_df["subset"] = "train"
# test_df["subset"] = "test"

# cell_size = 0.1  # degrés
# margin = 1 

########### CS2 ###########
path = "data/database_split.csv"

train_df = p.read_csv(path)[["survey_id", "longitude", "latitude", "subset"]]
train_df = train_df.rename(columns={"survey_id": "surveyId", "longitude": "lon", "latitude": "lat"})
test_df = train_df[train_df["subset"] == "test"]
train_df = train_df[train_df["subset"] == "train"]

cell_size = 0.5  # degrés
margin = 2.0 

###########################

df = p.concat([train_df, test_df], ignore_index=True)
df = df.dropna(subset=["lon", "lat"])  # sécurité si des surveyId n'ont pas de coordonnées

geometry = [Point(xy) for xy in zip(df["lon"], df["lat"])]
points = gp.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")




def square_bounds(bounds):
    xmin, ymin, xmax, ymax = bounds
    lon_range = xmax - xmin
    lat_range = ymax - ymin
    mean_lat = (ymin + ymax) / 2

    lon_range_m = lon_range * np.cos(np.radians(mean_lat))
    lat_range_m = lat_range

    if lon_range_m > lat_range_m:
        pad = (lon_range_m - lat_range_m) / 2
        ymin -= pad
        ymax += pad
    else:
        pad = (lat_range_m - lon_range_m) / (2 * np.cos(np.radians(mean_lat)))
        xmin -= pad
        xmax += pad

    return (xmin, ymin, xmax, ymax)

def make_grid(bounds, cell_size):
    xmin, ymin, xmax, ymax = bounds
    rows = np.arange(ymin, ymax + cell_size, cell_size)
    cols = np.arange(xmin, xmax + cell_size, cell_size)

    cells = []
    for x in cols:
        for y in rows:
            cells.append(box(x, y, x + cell_size, y + cell_size))

    grid = gp.GeoDataFrame({"geometry": cells}, crs="EPSG:4326")
    grid["cell"] = range(len(grid))
    return grid

xmin, ymin, xmax, ymax = points.total_bounds
bounds = (xmin - margin, ymin - margin, xmax + margin, ymax + margin)
bounds = square_bounds(bounds)
grid = make_grid(bounds, cell_size)



joined = gp.sjoin(points, grid, how="left", predicate="within")

def classify(cell_id, joined):
    sub = joined.loc[joined["cell"] == cell_id, "subset"]
    has_test = (sub == "test").any()
    has_train = (sub == "train").any()
    if has_test:
        return "Test"
    elif has_train:
        return "Train"
    else:
        return "Aucun"

grid["subset"] = grid["cell"].apply(lambda c: classify(c, joined))


palette = {
    "Aucun": "#F8F8F8",
    "Train": "#FFB22C",
    "Test": "#40514E",
}
grid["color"] = grid["subset"].map(palette)


fig, ax = plt.subplots(figsize=(6, 6))

m = Basemap(
    projection="merc",
    llcrnrlon=bounds[0], llcrnrlat=bounds[1],
    urcrnrlon=bounds[2], urcrnrlat=bounds[3],
    resolution="i",
    ax=ax,
)

m.drawcoastlines(color="black", linewidth=0.3)
m.drawcountries(color="black", linewidth=0.3)
m.fillcontinents(color="#FFFFFF", lake_color="#FFFFFF", zorder=0)
m.drawmapboundary(fill_color="#FFFFFF")


patches = []
colors = []

for _, row in grid.iterrows():
    coords = np.array(row.geometry.exterior.coords)
    x, y = m(coords[:, 0], coords[:, 1])
    xy = np.column_stack([x, y])
    patches.append(MplPolygon(xy, closed=True))
    colors.append(row["color"])

pc = PatchCollection(patches, facecolor=colors, edgecolor=colors, linewidths=0.3, zorder=1)
ax.add_collection(pc)


legend_order = ["Train", "Test"]
legend_elements = [
    Patch(facecolor=palette[k], edgecolor=palette[k], label=k)
    for k in legend_order
    if k in grid["subset"].unique()
]
ax.legend(
    handles=legend_elements,
    title="Subset",
    loc="lower right",
    frameon=True,
    facecolor= palette["Aucun"],
    edgecolor= "black",
    fontsize=12,
    title_fontsize=18,
)

plt.tight_layout()
plt.savefig("figures/split-data.png", dpi=300, bbox_inches="tight")
plt.close()