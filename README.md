AI agent: Jacky <br>
Author: Jackson Wu (jackson911007@gmail.com )<br>
Date: Jan. 14 2026 <br>

* Packages require: <br>
(1) camelot <br>
(2) pdfplumber <br>
(3) google.generativeai <br>
(4) arxiv <br>

Here we built a automated literature review pipeline: <br>
* There are two main scripies in the "Code" folder web_scrapping_arxiv.py and Jacky.py <br>
* web_scrapping_arxiv.py uses "arxiv" package developed by Cornell University Library (https://pypi.org/project/arxiv/) to do crawling from arxiv database <br>
* Jacky.py is the code to do pdf information extraction using Google Gemini <br>
<Please first visit "Gemini api studio" to create a api key> <br>

Code explanation: <br>
* web_scrapping_arxiv.py did web crawling by defining prompt to search on arvix website. It then downloads all related literatures and stored in user-defined folder <br>
* Jacky separates to two main parts: <br>
(1) Read in pdf file --> sparse the text and extract the text from table ---> combine all the table and contect information <br>
(2) Apply Gemini to digest the text message. To do so, you need to define clear message (prompt) to Gemini and design the output <br>
* Jacky outputs a csv file that collects the information extracted from pdf based on user-defined prompt <br>
