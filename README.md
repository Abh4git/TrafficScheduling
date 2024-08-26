# TrafficScheduling
Traffic Scheduling for Time Sensitive Networks  

Run the program schedulegenwithdeadline.py

![Resulting Schedule](schedule8Links.JPG)

A brief description

Scheduling being an NPHard problem many methods have been applied.
Genetic Algorithms, Metaheuristics all have found its advantages.
I found it easier with Genetic algorithms as finally all you do is
try creating different populations and check the fitness (whether rules or constraints are met)
more relatable. This is the approach in Genetic. 

Inline image

An initial flow sequence is provided by JSON for 10 Links
![image](https://github.com/user-attachments/assets/d3a073fb-2120-4026-ae8b-0cbda77eb37f)

The Traffic Model is here. The flows (4) are described in code here
![image](https://github.com/user-attachments/assets/40209bdb-b804-4f45-a6ca-2561cb66fd51) 

Main method  
![image](https://github.com/user-attachments/assets/ee561a0a-76f6-4bb7-bb47-996def607bca)  
Include calls to prepare initial polpulation  
Multiple Iterations involved to find the best  
![image](https://github.com/user-attachments/assets/b708fbb6-0ed8-41dd-a99d-8bfe5950f57b)  
Perform mutations (rearrangements) and check fitness  
![image](https://github.com/user-attachments/assets/85ae25f5-dd4d-448a-977c-1df203865d2f)  
Identifying the best schedule by checking makespan. The Shortest Makespan is considered best  
![image](https://github.com/user-attachments/assets/b5d1b91e-fecb-40ac-bdfe-3fdf63b86562)  






