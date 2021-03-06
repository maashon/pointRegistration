# PointRegistration

In this project, ICP and Tr-ICP are implemented to obtain the pointCload registration.
The code take two different pointCloads and tries to find the proper transformation to register the two partially-overlaping point cloads.

Different pointCloads are rotated using  saveRotated() function. Also Gaussian noise with 0 mean and standard deviation of 0.1,0.05 and 0.2 is added to one of the point cloads to observe the effect of noise in the corresponding algorithm.Also Rotation and Gaussian noise were applied to a point cload to observe the effect of both factors at the same time.


Note:A python script was utilized to break a single point cload to two different overlapping Pointcloads which was ran on Google Colab service to speed up the process. the script can be found with the name "pointCload.py" in the files 

Here are some screenshots of the pointCloads:

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
The program provides us with the error in each iteration and the runtime of the Algorithm(ICP or tr-ICP) at the end of each run.

Note: the following charts are obtained Usiong Excel.The runtime console results and the charts are available in Results.xlsx file.

1- the decrease of the Error in ICP during iterations:

![image](https://user-images.githubusercontent.com/72257286/151469849-668c0c63-af44-48c7-9b2c-1bf743d2d055.png)


2- The decrease of the Error in tr-ICP during iterations:


![image](https://user-images.githubusercontent.com/72257286/151469974-4ddbf290-8c85-4a69-acbf-9a302701f037.png)


3- Errors Of ICP algorithms for different point-cload pairs:

![image](https://user-images.githubusercontent.com/72257286/151470135-2fae530f-94fa-4d44-a2ac-1392fa475a38.png)


4-ICP Rintime For Different point-cload pairs:

![image](https://user-images.githubusercontent.com/72257286/151470197-a879dcb2-cc76-4059-964a-9b691b415070.png)


5- Errors Of Tr-ICP algorithms for different point-cload pairs:

![image](https://user-images.githubusercontent.com/72257286/151470237-12a86060-2d35-4e98-92c8-0faee1e57445.png)


6-Tr-ICP Rintime For Different point-cload pairs:

![image](https://user-images.githubusercontent.com/72257286/151470292-a91a0dba-7a43-4629-b0d5-828d1e0e9897.png)

7- ICP Runtime Bar chart

![image](https://user-images.githubusercontent.com/72257286/151487372-6e7b8671-cc11-4916-b851-f4d921b9f2d8.png)



8- Tr-ICP Runtime bar chart

![image](https://user-images.githubusercontent.com/72257286/151487415-4b27916f-4c13-4b52-a7c6-5d23da929c7b.png)


9-ICP Error Bar chart

![image](https://user-images.githubusercontent.com/72257286/151487453-373357a7-7537-4df2-9a02-8a02c1996a01.png)


10- Tr_ICP Bar cahrt

![image](https://user-images.githubusercontent.com/72257286/151487486-42c52676-1ea0-41d5-9944-c507a8fb0eb9.png)



11- Average Runtime and Errors for Ech algorithm:

![image](https://user-images.githubusercontent.com/72257286/151487531-c7aa8b94-5a6e-44df-8edd-1bdffdcfd078.png)


# conclustions

1- According to the runtime of the Algorithms, we observe that the ICP is slower than TR-ICP . the main reason is that Tr-ICP trims the point cloads each iteratoin and hence, deals with relatively less data than ICP which results to a significant speedup.

2- the Tr-ICP algorithm is more robust and performs better on the same set of Point cloads. It does a noticable better job in comparison to ICP.

3- Adding Gaussian noise to the Point cloads gives the model a hard time to do the registratoin as it adds some outliers to the pointcload. If we take a look and runtime of the algorithms, the gaussian added noise case takes significantly more time (of course more iteratoins) to converge.















