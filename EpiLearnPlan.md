# Plan for EpiLearn

1. Install EpiLearn and run toy example
    - Deliverables:
        1. Plot of toy spatial graph
        2. Summary statistics of toy features
        3. Verbal summary, can we do this with our data?

2. Convert our model to graph form
    - Deliverables:
        1. Function to go from raw whooping crane data to geospatial python object (probably geodataframe)
            - Should allow for different grid sizes, temporal binnings
        3. Function to go from geospatial object to spatial graph usable by epilearn
        4. Plot of spatial graph for a whooping crane grid
        5.  Function to go from raw opioid data to geospatial object (this might take Kyle's involvement, we could probably start from existing GDFs) 
        6. Plot of spatial graph for MA, cook county

3. Run EpiLearn models on our graphs
    - Deliverables:
        1. Way to train test split
        2. Function that accepts our geospatial grid defined by step 2, train test splits in some easy and clear way, and trains a simple epilearn spatiotemporal model, such as GCN, to make predictions.
        3. Function to take GCN predictions and calculate deterministic BPR
        4. Apples-to-apples report of BPR on a whooping crane dataset.
