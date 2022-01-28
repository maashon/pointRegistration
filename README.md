# pointRegistration

In this project, ICP and Tr-Icp are implemented to obtain the point cload registration.
the code take two different point cloads and tries to find the proper transformation to register the two point partially-overlaping point cloads.

Different point cloads are rotated using  saveRotated() functoin. also Gaussian noise with 0 mean and standard deviation of 0.1 is added to one of the point cloads to observe the effect of noise in the corresponding algorithm.

Here are some screenshots of the point cloads:

1-Two point cloads of Aloe (obtained from the first Assignment output) together:

![Aloe together](https://user-images.githubusercontent.com/72257286/151468725-723c5bd1-016b-4094-9b3f-51d43edd8554.png)


2-Two point cloads of Baby (obtained from the first Assignment output) together:


![Baby together](https://user-images.githubusercontent.com/72257286/151468764-7ca5ba36-2a1b-40d4-96e7-f2e9c34380bb.png)


3- Two Fountain point cloads provided in the course website together:

![Fountain together](https://user-images.githubusercontent.com/72257286/151468805-bcdee56a-38a4-49e3-9201-0c14620bede1.png)


4-Fountain_a with and without Gaussian noise together :

![Fountain_a And Gaussian noisy Fountain_a ](https://user-images.githubusercontent.com/72257286/151469308-09214a5d-3af1-40fb-acfd-e3acdef480d9.png)

5- Aloe_1 and Aloe_1 rotated version together:

![Aloe1 and 15 Degree Rotated ](https://user-images.githubusercontent.com/72257286/151469372-39c4fa07-15e0-42fa-a1d3-7afff04fb673.png)


# ICP OUTPUTS:

1-Fountain simple (RGB)

![Fountain simple RGB ICP ](https://user-images.githubusercontent.com/72257286/151468855-0f23ad18-86f9-4ec9-af04-5780d16f2817.png)

2- Baby with Gaussian noise added to first point cload (single color)

![Baby ICP with Gaussin noise](https://user-images.githubusercontent.com/72257286/151469078-27ce5760-5bf0-46ec-91a8-3550872e8bba.png)



# Tr-ICP OUTPUTS :

1-Aloe Simple Tr-ICP output (RGB):


![Aloe Tr-ICP RGB](https://user-images.githubusercontent.com/72257286/151469133-90349664-62b7-4a9b-ac57-820206a32814.png)

2-Baby 10 degree Roteted (RGB):

![baby 10 degree rotaed RGB TR-ICp  ](https://user-images.githubusercontent.com/72257286/151469175-88d8b9a9-d53d-40a9-b14b-bd0f9a0db5dd.png)


3-Aloe TR-ICP single color outout:


![Aloe Tr-ICP normal ](https://user-images.githubusercontent.com/72257286/151469524-5e5b4f40-14b9-4cc7-bf55-5ddc61a68bff.png)



# charts and plots:



