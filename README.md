# city-to-scan-to-city

## What?

This pipeline was developed to conduct experiments that investigate how point cloud properties (e.g., density, accuracy, completeness) influence the quality of 3D building reconstruction. It served as the main tool of analysis for my master's thesis project. The final thesis on *The Impact of Point Cloud Density and Accuracy on the Quality of 3D Building Reconstruction* is available [here](resources/Master's%20Thesis%20Florian%20Faltermeier%20v1.1.pdf).

## Why?

From a technical perspective, the quality of 3D building reconstruction depends on many factors related, for instance, to the reconstruction method and the input data quality. Much work has been dedicated to the development and comparison of different reconstruction methods, but few studies have explored the significance of the input data quality. The results of this project help to close this gap. Moreover, a pipeline such as this can also provide insights that support the improvement of reconstruction methods.

From an economic point of view, the acquisition of point cloud data for the purpose of 3D building reconstruction is costly. Knowledge of the quantitative relationship of point cloud properties and reconstruction quality facilitates a well-informed survey design to optimize the latter with respect to (the acquisition cost of) the former, or to identify levels of the former to meet requirements in the latter.

## How?

![A figure explaining the basic approach of the pipeline](resources/basic_idea_150dpi.png)

The above figure illustrates the basic approach: Based on an input dataset of 3D building models, airborne laser scanning (ALS) is simulated to generate point clouds with properties varying along two dimensions. Subsequently, a new dataset of 3D building models is reconstructed from each point cloud. Each output dataset has a one-to-one correspondence between the output and the input buildings. This enables a quantitative evaluation of the reconstruction quality for each building model with respect to the reference model. Provided the dataset is sufficiently large, the evaluation can distinguish between subsets of buildings with different geometric or, if available, semantic properties.

In its current shape, the pipeline is designed to work with building models from the 3D cadastre of the Netherlands ([3DBAG](https://3dbag.nl)), but building models from any source could be used in principle. The ALS simulation is performed with [HELIOS++](https://github.com/3dgeo-heidelberg/helios). In this project, the software [Geoflow](https://github.com/geoflow3d) was used for the 3D reconstruction step, but again, any reconstruction software could be used in principle, which is one of many opportunities to extend the scope of the project and obtain further insights.
