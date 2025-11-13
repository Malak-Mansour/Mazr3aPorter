# Mazr3aPorter


## Tasks division:

#### (Swapnil, Grik) Gazebo simulation of the farm and mobile robot

#### (Rawda) Drone simulation
- https://github.com/Robotisim/drones_ROS2/wiki/Project-%231:-Follow-Ground-Robot
- https://github.com/Robotisim/drones_ROS2?tab=readme-ov-file

#### (Malak, Asma, Tunpitcha) Ground-robot algorithm
- https://www.linkedin.com/posts/michael-mcguire-a9b34b136_ai-robotics-autonomy-ugcPost-7392137760654123008-IFWC?utm_source=social_share_send&utm_medium=android_app&rcm=ACoAACmrc18B-M84jPjbGrVwi8RcT2RLqvpJE98&utm_campaign=copy_link

#### (Mohamed) App and Presentation



## Technical explanation/theory
- Unlike discrete planners (A*, D*, RRT), the potential field is continuous.

- Sensors:
    LiDAR: position of obstacles to avoid with potential field repulsion (helpful for dynamic obstacles)
    Drone camera: map and obstacles that cant be seen by the LiDAR (eg: puddle of water)
    GPS at the warehouse and robot
    Human's phone has GPS

## Resources
- Image sources:
    date palm: https://www.freepik.com/premium-ai-image/palm-trees-top-view-isolated-date-palm-tree-white-background-with-green-leaves_342932627.htm
    warehouse: https://www.freepik.com/premium-photo/aerial-top-view-warehouses-hangars-near-industrial-factory-zone-f_7866141.htm
    husky robot: https://www.dreamstime.com/illustration/warehouse-top-view.html
    cat: https://www.123rf.com/photo_129929636_red-plump-cat-is-sitting-on-the-street-top-view.html
    dog: https://commons.wikimedia.org/wiki/File:Kattakal_alias_kattai_dog_nagapattinam_indian_breed_top_view.jpg
    rabbit: 
        https://www.shutterstock.com/search/bunny-on-the-grass?page=18 
        https://www.123rf.com/photo_11418585_red-rabbit-in-grass-top-view.html 
    human: https://www.alamy.com/man-farmer-working-in-vegetable-garden-collects-a-cucumber-top-view-and-copy-space-template-image218818773.html
    dates: https://pngtree.com/freebackground/top-view-of-fresh-and-authentic-dates-from-arabic-date-palm-tree_13622393.html 


- Date types classification
    1. https://www.researchgate.net/publication/382588066_Date_Fruit_Detection_and_Classification_based_on…
    2. https://www.researchgate.net/publication/351440988_Dataset_for_localization_and_classification_of_M…
    3. https://www.mdpi.com/2079-9292/12/3/665 Their dataset: http://doi.org/10.5281/zenodo.4639543
    4. (BEST) Model and dataset available (maturity level and date type classification): https://www.kaggle.com/code/ulaelg/date-fruit-predictions
