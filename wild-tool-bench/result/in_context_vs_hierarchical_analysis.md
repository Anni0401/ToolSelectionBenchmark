# WTB: In-context vs. hierarchical tool selection

> Automatisch erzeugter Vergleich der vorhandenen WTB-Läufe. Korrektheit und Token-/Latenzwerte stammen aus den `Wild-Tool-Bench_result.jsonl`-Dateien; Selection-Logs werden separat ausgewertet.

## Executive Summary

- Vergleichbare Tasks: **944** (In-context insgesamt: 948, hierarchical insgesamt: 1020).
- Korrekte Tasks: **422/948 (44.5%)** vs. **436/1020 (42.7%)**.
- Optimale Tasks: **383/948 (40.4%)** vs. **393/1020 (38.5%)**.
- `optimal` ist die vom WTB-Evaluator markierte optimale Aktionsauswahl; es ist nicht dasselbe wie eine bloß syntaktisch gültige Tool-Call-Antwort.

## Gesamtmetriken

| Metrik | In-context | Hierarchical | Differenz (hier. − in-context) |
|---|---:|---:|---:|
| Tasks | 948 | 1020 | 72 |
| Correct tasks | 422 | 436 | 14 |
| Optimal paths/tasks | 383 | 393 | 10 |
| Exact tool-call paths | 354 | 382 | 28 |
| Predicted tool calls | 776 | 702 | -74 |
| Gold tool calls | 1688 | 1794 | 106 |
| Positionally matched calls | 355 | 347 | -8 |
| Input tokens | 138010866 | 2247133 | -135763733 |
| Output tokens | 824375 | 805628 | -18747 |
| Total tokens | 138835241 | 3052761 | -135782480 |
| Ø tokens / task | 146450.68 | 2992.90 | -143457.77 |
| Median tokens / task | 89975.50 | 2185.50 | -87790.00 |
| LLM latency (s) | 51527.38 | 42415.11 | -9112.27 |
| Ø latency / task (s) | 54.35 | 41.58 | -12.77 |
| Median latency / task (s) | 26.95 | 26.77 | -0.19 |
| Wall time (s) | 51543.28 | 42430.04 | -9113.24 |
| Tool-call steps | 1583 | 1690 | 107 |
| Ø steps / task | 1.67 | 1.66 | -0.01 |

### Selection-Overhead

| Metrik | In-context | Hierarchical |
|---|---:|---:|
| Selection log records | 1 | 1021 |
| Ø available tools | 1283.0 | 1283.0 |
| Ø selected tools | 1283.0 | 5.4 |
| Median selected tools | 1283.0 | 3.0 |
| Ø reduction | 0.0% | 99.6% |
| Empty selections | 0 | 24 |

## Paarweiser Task-Vergleich

| Ergebnis | Anzahl | Anteil der gemeinsamen Tasks |
|---|---:|---:|
| Beide korrekt | 313 | 33.2% |
| Nur in-context korrekt | 108 | 11.4% |
| Nur hierarchical korrekt | 86 | 9.1% |
| Beide inkorrekt | 437 | 46.3% |

### Tasks mit deutlichem Unterschied

Die folgenden Tabellen listen die paarweisen Fälle, in denen genau eine Strategie korrekt bzw. optimal war.

| Task | In-context | Hierarchical | Text (gekürzt) |
|---|---|---|---|
| wild_tool_bench_0 / 0 | error / optimal=False | correct / optimal=True | I want to travel to other places this weekend. Please help me check the weather in Chicago for the two days of the weekend. |
| wild_tool_bench_1 / 1 | correct / optimal=True | error / optimal=False | Artificial Intelligence Security in the Past Three Months |
| wild_tool_bench_1 / 2 | error / optimal=False | correct / optimal=True | Have there been any relevant academic reports or discussions on the above two topics in Switzerland in the past two months? |
| wild_tool_bench_10 / 2 | error / optimal=False | correct / optimal=True | I don't want it anymore. Please help me return it again. |
| wild_tool_bench_102 / 3 | error / optimal=False | correct / optimal=True | The original name and its age distribution among different LGBT genders in the United States. |
| wild_tool_bench_103 / 0 | error / optimal=False | correct / optimal=False | I need to prepare teaching materials about the word'resilience'. Please help me obtain the detailed definition of this word, examples of its usage in sentences, its pronunciation,  |
| wild_tool_bench_103 / 1 | error / optimal=False | correct / optimal=False | You also need to prepare an 'Inscrutable'. |
| wild_tool_bench_104 / 2 | correct / optimal=True | error / optimal=False | I want to see the pictures. |
| wild_tool_bench_105 / 3 | correct / optimal=True | error / optimal=False | Search for the hosts of 'example.net' in the location of the last IP. |
| wild_tool_bench_106 / 2 | correct / optimal=True | error / optimal=False | What about the corresponding historical data in the US? |
| wild_tool_bench_107 / 0 | error / optimal=False | correct / optimal=False | I want to know the scorecard of the game with ID 456, including detailed information. Also, please provide the batting average statistics of the player with ID 789 in the 2022 seas |
| wild_tool_bench_108 / 2 | error / optimal=False | correct / optimal=True | What about the altitude? |
| wild_tool_bench_108 / 3 | error / optimal=False | correct / optimal=True | Help me query the information of 'example.com' again, with the same requirements as at the beginning. |
| wild_tool_bench_11 / 0 | error / optimal=False | correct / optimal=True | Please tell me the list of holidays in the United States the year before last. |
| wild_tool_bench_11 / 1 | correct / optimal=True | error / optimal=False | China |
| wild_tool_bench_110 / 3 | error / optimal=False | correct / optimal=True | Check the 5-day data of the link that comes after the link with the second-highest click-through rate. |
| wild_tool_bench_111 / 1 | correct / optimal=False | error / optimal=False | What about the data from the previous year? |
| wild_tool_bench_112 / 2 | correct / optimal=True | error / optimal=False | What if the location is 40.7128, 74.0060? |
| wild_tool_bench_114 / 1 | error / optimal=False | correct / optimal=True | Get the geographical coordinates of the third step of the route. |
| wild_tool_bench_115 / 1 | correct / optimal=True | error / optimal=False | Population status. |
| wild_tool_bench_116 / 0 | correct / optimal=False | error / optimal=False | I need to view all the gluten-free products I purchased at Kroger in the past three months and find the nearest Kroger store so that I can repurchase them. |
| wild_tool_bench_116 / 1 | error / optimal=False | correct / optimal=True | Help me search for similar products in Target. |
| wild_tool_bench_118 / 1 | error / optimal=False | correct / optimal=True | View the cafes around here. |
| wild_tool_bench_12 / 0 | correct / optimal=True | error / optimal=False | Can you help me search for some open data sets about the economy of Cyprus last year? |
| wild_tool_bench_12 / 2 | correct / optimal=False | error / optimal=False | Download the other two as well. Then, I also want to know what has been recently released in Germany and Sweden. |
| wild_tool_bench_122 / 3 | error / optimal=False | correct / optimal=True | Help me plan how to walk from my uncle's house to my home again. |
| wild_tool_bench_123 / 1 | correct / optimal=True | error / optimal=False | Could you show the detailed information of this position? |
| wild_tool_bench_125 / 1 | error / optimal=False | correct / optimal=False | Sell what I just bought and buy 5 ETH again. |
| wild_tool_bench_125 / 3 | error / optimal=False | correct / optimal=False | Get some more LTC. The quantity is twice the quantity of the first-round currency transaction plus half of the quantity of the second-round currency transaction. Also, I need ETC.  |
| wild_tool_bench_126 / 1 | correct / optimal=True | error / optimal=False | How do you pronounce it? |
| wild_tool_bench_128 / 0 | correct / optimal=True | error / optimal=False | I want to announce the upcoming team-building activity next month in any chat group of Wang Wu. |
| wild_tool_bench_128 / 1 | correct / optimal=True | error / optimal=False | Help me find out which other things he still has access to. |
| wild_tool_bench_129 / 2 | error / optimal=False | correct / optimal=True | Help me check the weather conditions near the airport in this place |
| wild_tool_bench_13 / 1 | correct / optimal=False | error / optimal=False | Rock, Folk, Electronic, Easy Listening |
| wild_tool_bench_130 / 0 | correct / optimal=True | error / optimal=False | Please help me update the email address of the user with ID 12345. |
| wild_tool_bench_131 / 3 | error / optimal=False | correct / optimal=False | Update the following two sentences into his famous quotes: 1. A program is like a poem. The less you write, the more you express. 2. Debugging code is more difficult than writing c |
| wild_tool_bench_134 / 2 | error / optimal=False | correct / optimal=True | By the way, what impact do these foods she ate have on our weight control? Could you give me their nutritional information? |
| wild_tool_bench_136 / 2 | correct / optimal=True | error / optimal=False | Help me purchase 100,000 yen and pounds sterling corresponding to this currency respectively. |
| wild_tool_bench_137 / 0 | error / optimal=False | correct / optimal=True | The corporate legal department has requested a report that needs to include the current status and bibliographic data of all our company's patents for the annual audit. |
| wild_tool_bench_138 / 1 | correct / optimal=False | error / optimal=False | The first one in the currency pair. Check how much I have now. Oh, and also for BTC. |
| wild_tool_bench_138 / 3 | correct / optimal=True | error / optimal=False | I just bought some cryptocurrencies. Please buy some more for me. |
| wild_tool_bench_139 / 0 | correct / optimal=True | error / optimal=False | Our team is developing a strategy game. Now we need to add a mod to test new features. The version of the mod can be set to the default first, but the game ID and the name of the m |
| wild_tool_bench_139 / 3 | correct / optimal=True | error / optimal=False | Okay, now I want to update the status of one of them to inactive |
| wild_tool_bench_14 / 0 | correct / optimal=True | error / optimal=False | Please echo the content object, where the text content is 'Test text' and it contains a number 123. |
| wild_tool_bench_141 / 2 | error / optimal=False | correct / optimal=True | Please help me check the statistics of me and my friend in Call of Duty (smzh_007) again |
| wild_tool_bench_141 / 3 | correct / optimal=True | error / optimal=False | Help me get the kill leaderboard (kill_board) of the game at the beginning. |
| wild_tool_bench_142 / 1 | correct / optimal=False | error / optimal=False | What about 192.168.15.24, 172.16.254.1 and 203.0.113.5? |
| wild_tool_bench_143 / 0 | correct / optimal=True | error / optimal=False | As a researcher, I am studying medical reports from different countries. Can you help me identify the professional terms in these reports? |
| wild_tool_bench_143 / 3 | correct / optimal=True | error / optimal=False | Which article does the latter part of the text come from? |
| wild_tool_bench_145 / 1 | correct / optimal=False | error / optimal=False | What about Waiting For Love and Lose Yourself? The ids are Avicii2028 and Avicii2029 |
| wild_tool_bench_146 / 0 | correct / optimal=True | error / optimal=False | Please help me check the server location of this online store. |
| wild_tool_bench_146 / 1 | error / optimal=False | correct / optimal=True | What news has happened in this place recently? |
| wild_tool_bench_15 / 2 | error / optimal=False | correct / optimal=True | What about the complementary color of the grayscale version? |
| wild_tool_bench_153 / 0 | correct / optimal=True | error / optimal=False | I need an animation to display my company name. The font size should be larger. |
| wild_tool_bench_153 / 1 | correct / optimal=True | error / optimal=False | Don't add Technology Co., Ltd. |
| wild_tool_bench_153 / 2 | error / optimal=False | correct / optimal=True | Set the typing speed to 60 words per second |
| wild_tool_bench_153 / 3 | correct / optimal=True | error / optimal=False | Take a look at the animation settings |
| wild_tool_bench_155 / 0 | correct / optimal=True | error / optimal=False | Can you help me check the weather yesterday? |
| wild_tool_bench_155 / 2 | correct / optimal=True | error / optimal=False | I'm going abroad for a trip next week from the place where I was yesterday. Can you help me check the weather abroad next week? |
| wild_tool_bench_157 / 2 | error / optimal=False | correct / optimal=False | Generate another 90-word one for me. In addition, I also need a demonstration text with 8 sentences. |
| wild_tool_bench_159 / 3 | error / optimal=False | correct / optimal=True | Which team does the player I first mentioned belong to? |
| wild_tool_bench_16 / 0 | error / optimal=False | correct / optimal=True | I want to know if there are any mentions of 'weather' in the real-time streamed tweets near New York City (longitude: -73.9857, latitude: 40.7484). |
| wild_tool_bench_162 / 1 | correct / optimal=False | error / optimal=False | There are visitors coming to visit our school from Hangzhou today. Can you check the weather in the two places today? |
| wild_tool_bench_162 / 3 | error / optimal=False | correct / optimal=True | Oh, I remember. The name of the poem is 'Farewell in the Mountains' |
| wild_tool_bench_163 / 0 | correct / optimal=True | error / optimal=False | Help me search for some anime and manga works |
| wild_tool_bench_164 / 1 | correct / optimal=True | error / optimal=False | Search for the detailed information of the first one |
| wild_tool_bench_165 / 2 | correct / optimal=True | error / optimal=False | Check the latest messages to see if they were sent successfully |
| wild_tool_bench_168 / 1 | error / optimal=False | correct / optimal=True | Are there any other stations with wheelchair access? |
| wild_tool_bench_168 / 2 | correct / optimal=True | error / optimal=False | What are the train platforms at the terminal station to reach there? |
| wild_tool_bench_168 / 3 | correct / optimal=True | error / optimal=False | Check the train schedule there again |
| wild_tool_bench_170 / 0 | correct / optimal=True | error / optimal=False | I want to search for the songs of a singer |
| wild_tool_bench_170 / 2 | correct / optimal=False | error / optimal=False | Help me get the detailed information of this song. Besides, I also need the details of another song with the id of adele123. |
| wild_tool_bench_171 / 0 | correct / optimal=True | error / optimal=False | I need a new API key to buy BTC and ETH. |
| wild_tool_bench_172 / 3 | error / optimal=False | correct / optimal=True | I also want to know the quest information related to this equipment |
| wild_tool_bench_173 / 0 | correct / optimal=True | error / optimal=False | Can you help me obtain the OSGB36 coordinates of a specific mountain within the Lake District National Park in northern England? I'm planning a hiking trip. |
| wild_tool_bench_174 / 2 | correct / optimal=False | error / optimal=False | First display all my short URLs, and then delete the last 2. |
| wild_tool_bench_176 / 1 | correct / optimal=True | error / optimal=False | I want to know the real-time information about the highway in the first alarm message. |
| wild_tool_bench_176 / 3 | error / optimal=False | correct / optimal=True | How long will it take if I take the third highway? |
| wild_tool_bench_177 / 0 | correct / optimal=True | error / optimal=False | Please help me process a BTC/USDT transaction on Huobi. |
| wild_tool_bench_18 / 0 | error / optimal=False | correct / optimal=True | I would like to list the short links generated under my account on the sixth page, displaying 6 results per page. My User ID is user123. Could you please handle this for me? |
| wild_tool_bench_18 / 3 | correct / optimal=True | error / optimal=False | Fourth from the last and then add 28 days. |
| wild_tool_bench_180 / 3 | correct / optimal=True | error / optimal=False | I want to view the detailed content of the news that is two articles after the first one. |
| wild_tool_bench_181 / 3 | error / optimal=False | correct / optimal=False | What about the others? |
| wild_tool_bench_182 / 1 | correct / optimal=True | error / optimal=False | What specific green travel methods can help improve the situation? |
| wild_tool_bench_182 / 3 | error / optimal=False | correct / optimal=True | What about in a few days? |
| wild_tool_bench_185 / 1 | error / optimal=False | correct / optimal=False | Among those platforms you mentioned, which one has the lowest handling fee for trading BTC? |
| wild_tool_bench_185 / 2 | error / optimal=False | correct / optimal=True | Can you help me check the real-time price of this cryptocurrency? |
| wild_tool_bench_185 / 3 | error / optimal=False | correct / optimal=False | Has this coin had large fluctuations in the past two days? |
| wild_tool_bench_186 / 2 | correct / optimal=True | error / optimal=False | Oh no, I think I remembered the wrong email. The email is colleagues@example.com. |
| wild_tool_bench_187 / 2 | error / optimal=False | correct / optimal=False | I want to know the achievements and rewards related to the third mount and the third companion you mentioned. |
| wild_tool_bench_188 / 2 | correct / optimal=True | error / optimal=False | 17732221123, SMS. |
| wild_tool_bench_189 / 1 | correct / optimal=True | error / optimal=False | I know the highlights you mentioned! How about the Knight in it? What's the normal acquisition difficulty? |
| wild_tool_bench_19 / 0 | error / optimal=False | correct / optimal=True | Hello, please provide the current market data of XRP against JPY on Bitfinex. Thank you! |
| wild_tool_bench_191 / 2 | correct / optimal=True | error / optimal=False | Can you help me check who the shares belong to? |
| wild_tool_bench_195 / 2 | error / optimal=False | correct / optimal=True | I also want to improve myself in this way, but the link to this course is too long. Can you shorten the first one for me? |
| wild_tool_bench_196 / 3 | error / optimal=False | correct / optimal=True | What is the specific content? |
| wild_tool_bench_20 / 0 | error / optimal=False | correct / optimal=True | Please provide the latest global COVID-19 statistics. |
| wild_tool_bench_20 / 2 | error / optimal=False | correct / optimal=True | The latest virus spread maps for these regions. |
| wild_tool_bench_20 / 3 | correct / optimal=True | error / optimal=False | I also need to know the virus data of a country, preferably from last year to the present. |
| wild_tool_bench_202 / 3 | error / optimal=False | correct / optimal=True | Check again for the second author of the next category in the second round. No need for the table of contents, and no duplicates. |
| wild_tool_bench_208 / 2 | correct / optimal=True | error / optimal=False | Oh, I also want to know what I have saved. I remember seeing some works of others before. I liked them very much and saved some. |
| wild_tool_bench_21 / 0 | correct / optimal=True | error / optimal=False | Please help me find the GDP data of the United States in 2022 and return it in JSON format. |
| wild_tool_bench_21 / 1 | correct / optimal=True | error / optimal=False | What about China's? |
| wild_tool_bench_21 / 3 | correct / optimal=True | error / optimal=False | Please check again the GDP and unemployment rate of the country I inquired about in the first round this year. |
| wild_tool_bench_211 / 2 | error / optimal=False | correct / optimal=True | What are the voting statistics of the first of the two states that allocate electoral votes by congressional district? |
| wild_tool_bench_212 / 2 | correct / optimal=True | error / optimal=False | Summarize these contents |
| wild_tool_bench_213 / 0 | error / optimal=False | correct / optimal=True | I want a random string as a password. How many digits should be set to be more appropriate? |
| wild_tool_bench_217 / 1 | correct / optimal=False | error / optimal=False | What industrial facilities in the central areas of Los Angeles and Las Vegas, as well as the financial district of San Francisco, have emissions? |
| wild_tool_bench_22 / 0 | error / optimal=False | correct / optimal=True | As a Protestant pastor, I need to prepare sermons for the upcoming religious festivals. Can you provide some information? |
| wild_tool_bench_22 / 3 | correct / optimal=True | error / optimal=False | Which countries celebrate this festival? |
| wild_tool_bench_220 / 1 | correct / optimal=False | error / optimal=False | I have a photo here and need you to perform the above two types of processing. The encoding is: iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVO |
| wild_tool_bench_221 / 1 | correct / optimal=True | error / optimal=False | Can you generate one for me? The length should be 10. |
| wild_tool_bench_23 / 0 | correct / optimal=True | error / optimal=False | Help me look up the detailed information of the artist with ID 12345. |
| wild_tool_bench_230 / 3 | error / optimal=False | correct / optimal=True | Oh, I also want to know what tracks the character mentioned at the beginning plays? |
| wild_tool_bench_235 / 2 | correct / optimal=True | error / optimal=False | This method seems very reliable. Are there any examples of its application in practice? |
| wild_tool_bench_239 / 2 | correct / optimal=False | error / optimal=False | These climate data are very helpful! I have a friend who wants to invest. I also want to know about the economic activities last year, including GDP, employment, and production, es |
| wild_tool_bench_24 / 0 | correct / optimal=True | error / optimal=False | I need the top news headlines published between July 1 and July 31 last year. |
| wild_tool_bench_24 / 1 | error / optimal=False | correct / optimal=True | I would like to know the definition and introduction of the content covered in the last news article? |
| wild_tool_bench_24 / 2 | error / optimal=False | correct / optimal=True | Find me some more relevant news reports. |
| wild_tool_bench_241 / 3 | error / optimal=False | correct / optimal=True | Los Angeles, iPhone 16 Pro, MacBook Pro |
| wild_tool_bench_242 / 3 | correct / optimal=True | error / optimal=False | Great, please help me book the train ticket. |
| wild_tool_bench_243 / 0 | error / optimal=False | correct / optimal=True | I need to know the train schedule and delay information at The Hague Station on the same day next week, at 2:30 pm. |
| wild_tool_bench_244 / 1 | correct / optimal=True | error / optimal=False | I want to know the candidate data of one of the parties |
| wild_tool_bench_245 / 3 | error / optimal=False | correct / optimal=True | I don't want these languages. |
| wild_tool_bench_246 / 0 | correct / optimal=True | error / optimal=False | As an investor, I need to track the dynamics of the cryptocurrency market in real time, especially changes in market capitalization and trading volume. |
| wild_tool_bench_246 / 3 | error / optimal=False | correct / optimal=True | Oh no, I bought the wrong thing. Can you help me cancel it? |
| wild_tool_bench_247 / 1 | correct / optimal=False | error / optimal=False | Nagoya and Osaka |
| wild_tool_bench_248 / 0 | error / optimal=False | correct / optimal=True | Please help me change the text of this SVG. |
| wild_tool_bench_249 / 1 | correct / optimal=True | error / optimal=False | Help me find the definition of one of the synonyms. |
| wild_tool_bench_25 / 0 | correct / optimal=True | error / optimal=False | I want to know what self-guided tour guide books in Polish are available. Can you help me look them up? |
| wild_tool_bench_25 / 1 | correct / optimal=False | error / optimal=False | Can you help me find the detailed information of these books? |
| wild_tool_bench_255 / 0 | correct / optimal=True | error / optimal=False | Please assist me in obtaining the account information of the Twitter user with user ID 11223, especially the recently liked tweets, and please include all possible entity informati |
| wild_tool_bench_255 / 1 | correct / optimal=True | error / optimal=False | Find some more similar to the one on May 9th. |
| wild_tool_bench_255 / 3 | correct / optimal=True | error / optimal=False | Which people has this author followed? |
| wild_tool_bench_26 / 2 | error / optimal=False | correct / optimal=True | I want to know the latest reviews and ratings of this store. |
| wild_tool_bench_26 / 3 | correct / optimal=True | error / optimal=False | What about other information? |
| wild_tool_bench_27 / 3 | error / optimal=False | correct / optimal=True | Which of the restaurants you mentioned offer vegetarian options? |
| wild_tool_bench_28 / 1 | correct / optimal=True | error / optimal=False | Check the price of another currency on this day last year for me. |
| wild_tool_bench_31 / 1 | correct / optimal=True | error / optimal=False | Check the information of another player for me. |
| wild_tool_bench_31 / 3 | correct / optimal=True | error / optimal=False | I want to know who is currently on one of the teams |
| wild_tool_bench_33 / 1 | correct / optimal=True | error / optimal=False | Subscribe to one more for me |
| wild_tool_bench_33 / 2 | error / optimal=False | correct / optimal=True | One more |
| wild_tool_bench_34 / 0 | error / optimal=False | correct / optimal=True | I want to know how much CNY can be exchanged for 100.5 JPY. The result needs to be rounded. |
| wild_tool_bench_34 / 2 | correct / optimal=True | error / optimal=False | I want to know the exchange rates of one of the currencies against the US dollar and the euro. |
| wild_tool_bench_35 / 2 | correct / optimal=False | error / optimal=False | The detailed information of his other two latest orders on Rappi. Oh, and also for Wang Wu and Li Si. For them, I need the first two orders. |
| wild_tool_bench_38 / 1 | correct / optimal=True | error / optimal=False | Help me search for some in another field. |
| wild_tool_bench_39 / 0 | error / optimal=False | correct / optimal=True | Can you generate some random dates? The format can be 'YYYY/MM/DD'. |
| wild_tool_bench_39 / 2 | error / optimal=False | correct / optimal=False | Generate the same number of strings for me based on the number of digits in these dates, plus 2. But use symbols instead of digits. |
| wild_tool_bench_40 / 1 | correct / optimal=True | error / optimal=False | Check another one for me |
| wild_tool_bench_40 / 2 | correct / optimal=True | error / optimal=False | I want to update the contact information of one of them to 565-423-1111 |
| wild_tool_bench_40 / 3 | error / optimal=False | correct / optimal=False | Detailed information of other internal medicine physicians in the city where this healthcare provider is located. |
| wild_tool_bench_41 / 0 | correct / optimal=True | error / optimal=False | Please help me search for articles on machine learning written by John Doe. |
| wild_tool_bench_46 / 1 | correct / optimal=True | error / optimal=False | Give me some more from a certain author |
| wild_tool_bench_48 / 1 | correct / optimal=True | error / optimal=False | I also want to look up a word |
| wild_tool_bench_49 / 3 | error / optimal=False | correct / optimal=True | I want to obtain more images of one of them. |
| wild_tool_bench_5 / 1 | error / optimal=False | correct / optimal=True | What about Daxing Airport? |
| wild_tool_bench_56 / 0 | correct / optimal=True | error / optimal=False | As a marketing manager, I need to create a short link for the launch of our new product. The original link is https://www.example.com/product-launch. Please ensure to use a custom  |
| wild_tool_bench_58 / 0 | error / optimal=False | correct / optimal=True | Can you tell me the traffic alerts in Sedona, Arizona, and obtain the real-time road conditions and camera images of Highway 89A? |
| wild_tool_bench_58 / 1 | error / optimal=False | correct / optimal=True | I want to know the current traffic conditions. |
| wild_tool_bench_58 / 3 | correct / optimal=True | error / optimal=False | Help me retrieve the image of the camera at the beginning again. |
| wild_tool_bench_59 / 2 | error / optimal=False | correct / optimal=True | There are really many scenic spots near the walking route. Could you introduce them? |
| wild_tool_bench_6 / 1 | correct / optimal=True | error / optimal=False | Sell 10 more |
| wild_tool_bench_6 / 3 | error / optimal=False | correct / optimal=True | I also want to buy something else |
| wild_tool_bench_61 / 0 | error / optimal=False | correct / optimal=False | As an English teacher, I need to prepare teaching materials about the word 'innovation'. I need a detailed definition of this word, examples of its use in sentences, its pronunciat |
| wild_tool_bench_65 / 3 | error / optimal=False | correct / optimal=True | What about the candidates' situation? |
| wild_tool_bench_66 / 0 | error / optimal=False | correct / optimal=False | I want to know what the complementary colors of #FF5733 and #33FF57 are? |
| wild_tool_bench_67 / 1 | error / optimal=False | correct / optimal=True | Please help me track the current location of the first train. |
| wild_tool_bench_68 / 3 | error / optimal=False | correct / optimal=False | The detailed information of the IP addresses in the example code (including timezone and currency), and the postal code. |
| wild_tool_bench_69 / 2 | error / optimal=False | correct / optimal=True | Create another campaign using the link with the highest number of clicks above. |
| wild_tool_bench_7 / 1 | correct / optimal=True | error / optimal=False | What are the other provinces? |
| wild_tool_bench_70 / 1 | correct / optimal=False | error / optimal=False | Please help me query the situation of player 'JohnDoe' on the second task and the'smithing' skill score. |
| wild_tool_bench_71 / 2 | correct / optimal=True | error / optimal=False | What about programming? |
| wild_tool_bench_75 / 0 | error / optimal=False | correct / optimal=False | Can you help me look up the basic information and pictures of salmon? Additionally, I would like to know about the population status of salmon, especially the sustainability rating |
| wild_tool_bench_75 / 1 | correct / optimal=True | error / optimal=False | Check the nutritional value. |
| wild_tool_bench_78 / 1 | error / optimal=False | correct / optimal=True | I want to know the price and inventory situation if I need to purchase 100 of the second type of processors. |
| wild_tool_bench_8 / 2 | correct / optimal=True | error / optimal=False | Mark one more for me |
| wild_tool_bench_80 / 1 | correct / optimal=True | error / optimal=False | Help me retrieve the user information and see what she likes. |
| wild_tool_bench_81 / 0 | error / optimal=False | correct / optimal=False | I would like to know the GDP growth rates of China and the United States in 2023. Please help me query the data of these two countries. |
| wild_tool_bench_81 / 2 | correct / optimal=False | error / optimal=False | Check the data for the United States in the past four years again. |
| wild_tool_bench_82 / 0 | correct / optimal=False | error / optimal=False | Please help me synchronize the account with user ID 12345, and then create a new transaction. The account ID is 67890, the date is 2023-10-01, the amount is $100.50, the payee is ' |
| wild_tool_bench_83 / 0 | error / optimal=False | correct / optimal=True | I want to know all the songs of the movie 《Dil Se》 and download the MP3 file of one of them. |
| wild_tool_bench_83 / 1 | correct / optimal=True | error / optimal=False | Help me look up the detailed information of the first song. |
| wild_tool_bench_83 / 2 | error / optimal=False | correct / optimal=True | Can you help me get the link to the songs in the movie 'Dil Se'? |
| wild_tool_bench_84 / 1 | error / optimal=False | correct / optimal=True | Then apply for the first Software Development Engineer position. |
| wild_tool_bench_85 / 0 | error / optimal=False | correct / optimal=False | I want to find a recently popular song named 'Unstoppable' and get its detailed information as well as a high-quality MP3 file. |
| wild_tool_bench_85 / 2 | correct / optimal=True | error / optimal=False | The detailed information of the second song. |
| wild_tool_bench_88 / 1 | error / optimal=False | correct / optimal=True | Get the hourly data of the underlying asset cryptocurrency in the first pair on this day last year. |
| wild_tool_bench_88 / 2 | error / optimal=False | correct / optimal=True | Introduce the detailed information of the quoted assets. |
| wild_tool_bench_9 / 0 | error / optimal=False | correct / optimal=True | Help me check the news from BBC the day before yesterday. |
| wild_tool_bench_9 / 2 | correct / optimal=True | error / optimal=False | I want to obtain the metadata of one of them |
| wild_tool_bench_90 / 3 | correct / optimal=True | error / optimal=False | Calculate the address 51.38.73.100 with the starting subnet mask. |
| wild_tool_bench_94 / 3 | error / optimal=False | correct / optimal=True | Then obtain the library that is 20 meters around the third coffee shop. |
| wild_tool_bench_98 / 3 | correct / optimal=True | error / optimal=False | The proportion of the two in the global confirmed cases. |
| wild_tool_bench_99 / 3 | correct / optimal=True | error / optimal=False | Influenza virus. |

## Gemeinsame Schwächen

| Schwäche | In-context | Hierarchical |
|---|---:|---:|
| Beide inkorrekt | 437 | 46.3% |
| Beide nicht optimal | 495 | 52.4% |
| Beide mit mindestens einem Tool-Call | — | 44.7% |

### Fehler nach Task-Typ

| Task-Typ | Gemeinsame Tasks | Beide inkorrekt | In-context Fehler | Hierarchical Fehler |
|---|---:|---:|---:|---:|
| Chat | 232 | 28 (12.1%) | 50 (21.6%) | 34 (14.7%) |
| Clarify | 236 | 160 (67.8%) | 171 (72.5%) | 195 (82.6%) |
| Mixed Multi-Tool | 77 | 72 (93.5%) | 74 (96.1%) | 74 (96.1%) |
| Parallel Multi-Tool | 148 | 89 (60.1%) | 109 (73.6%) | 106 (71.6%) |
| Sequential Multi-Tool | 12 | 10 (83.3%) | 10 (83.3%) | 10 (83.3%) |
| Single-Tool | 239 | 78 (32.6%) | 109 (45.6%) | 126 (52.7%) |

## Methodik und Einschränkungen

- Die Runs sind nicht vollständig gleich groß: Es werden nur Tasks verglichen, die in beiden Result-Dateien vorhanden sind.
- Die in-context-Tool-Call-Logs enthalten keine `test_entry_id`/`task_idx`; deshalb werden Call-Details primär aus den Result-Inference-Logs gelesen und Selection-Logs nur aggregiert.
- `matched tool calls` werden gegen die WTB-Gold-Actionpfade positionsweise gematcht; bei mehreren Goldpfaden wird das beste Matching verwendet.
- Ein hoher Tokenwert im In-context-Lauf ist erwartbar, weil dort sehr viele Tools an das ausführende LLM übergeben werden. Das ist Overhead, aber nicht automatisch ein Qualitätsfehler.
