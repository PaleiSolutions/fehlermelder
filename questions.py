"""
Inline-Datensatz: kurze Faktenfragen mit eindeutiger, per Normalized-Match prüfbarer Antwort.

Format je Zeile:  TIER | DOMAIN | Frage | Alias1; Alias2; ...
  TIER   E = leicht, M = mittel, H = schwer   (Streuung, damit ein 7B-Modell grob 40–70 % trifft)
  DOMAIN gen = Allgemeinwissen, fin = Finanz/Banking

Alle Antworten wurden beim Erstellen geprüft; Fragen mit zeitabhängiger oder strittiger
Antwort (z. B. Hauptstadt Burundis, Zahl der Euro-Staaten) wurden bewusst weggelassen.
Wer Labels auditieren will: outputs/generations.csv enthält jede Modellantwort mit Label.
"""

DATA = r"""
E|gen|What is the capital of France?|Paris
E|gen|What is the capital of Japan?|Tokyo
E|gen|What is the chemical symbol for gold?|Au
E|gen|How many continents are there on Earth?|7;seven
E|gen|Which planet is known as the Red Planet?|Mars
E|gen|Who wrote 'Romeo and Juliet'?|William Shakespeare;Shakespeare
E|gen|What is the largest ocean on Earth?|Pacific Ocean;Pacific
E|gen|What is the boiling point of water in degrees Celsius at sea level?|100
E|gen|What gas do plants absorb from the atmosphere for photosynthesis?|carbon dioxide;CO2
E|gen|Who painted the Mona Lisa?|Leonardo da Vinci;da Vinci;Leonardo
E|gen|What is the largest planet in our solar system?|Jupiter
E|gen|In which country are the pyramids of Giza?|Egypt
E|gen|What is the chemical formula of water?|H2O
E|gen|What is the capital of Italy?|Rome
E|gen|How many sides does a hexagon have?|6;six
E|gen|What is the square root of 144?|12;twelve
E|gen|Which element has the atomic number 1?|hydrogen
E|gen|What is the currency of the United Kingdom?|pound sterling;pound;GBP
E|gen|Who was the first person to walk on the Moon?|Neil Armstrong;Armstrong
E|gen|What is the longest river in South America?|Amazon
E|gen|What is the capital of Germany?|Berlin
E|gen|What language is primarily spoken in Brazil?|Portuguese
E|gen|How many days are in a leap year?|366
E|gen|What is the hardest natural substance?|diamond
E|gen|Which animal is known as the King of the Jungle?|lion
E|gen|Who developed the theory of relativity?|Albert Einstein;Einstein
E|gen|What is the capital of Spain?|Madrid
E|gen|What is the freezing point of water in degrees Fahrenheit?|32
E|gen|What is the smallest prime number?|2;two
E|gen|In which city is the Eiffel Tower located?|Paris
E|gen|What is the capital of Canada?|Ottawa
E|gen|What is the capital of Australia?|Canberra
E|gen|How many legs does a spider have?|8;eight
E|gen|What is the main language spoken in Mexico?|Spanish
E|gen|Which organ pumps blood through the human body?|heart
E|gen|What is the chemical symbol for sodium?|Na
E|gen|Who composed the Fifth Symphony that begins with a famous four-note motif?|Ludwig van Beethoven;Beethoven
E|gen|What is the capital of Russia?|Moscow
E|gen|How many minutes are in an hour?|60;sixty
E|gen|What is the largest mammal?|blue whale
E|gen|What is the capital of China?|Beijing;Peking
E|gen|Which planet is closest to the Sun?|Mercury
E|gen|What is 7 multiplied by 8?|56
E|gen|What is the tallest mountain in the world above sea level?|Mount Everest;Everest
E|gen|What is the capital of Egypt?|Cairo
E|gen|Who wrote 'Hamlet'?|William Shakespeare;Shakespeare
E|gen|What colour do you get by mixing blue and yellow paint?|green
E|gen|What is the capital of the United States?|Washington DC;Washington
E|gen|What is the largest country in the world by area?|Russia
E|gen|On which continent is Kenya located?|Africa
E|gen|What is the chemical symbol for iron?|Fe
E|gen|What instrument has 88 keys in its standard form?|piano
E|gen|What is the capital of Greece?|Athens
E|gen|Which gas makes up most of Earth's atmosphere?|nitrogen
E|gen|What is the capital of the Netherlands?|Amsterdam
E|gen|Who is the author of the Harry Potter books?|J K Rowling;Rowling
E|gen|How many players does one team have on the field in a football (soccer) match?|11;eleven
E|gen|What is the capital of India?|New Delhi;Delhi
E|gen|Which country gave the Statue of Liberty to the United States?|France
E|gen|What is the capital of Portugal?|Lisbon
E|gen|What is the primary ingredient in guacamole?|avocado
E|gen|How many hours are in a day?|24
E|gen|What is the capital of Turkey?|Ankara
E|gen|Who discovered penicillin?|Alexander Fleming;Fleming
E|gen|Which planet has the most prominent ring system?|Saturn
E|gen|What is the capital of Sweden?|Stockholm
E|gen|What is the capital of Norway?|Oslo
E|gen|What is the capital of Poland?|Warsaw
E|gen|Which metal is liquid at room temperature?|mercury
E|gen|What is the capital of Austria?|Vienna
E|gen|What is the name of the galaxy that contains our solar system?|Milky Way
E|gen|What is 15 squared?|225
E|gen|Which ocean lies between Africa and Australia?|Indian Ocean;Indian
E|fin|What is the currency of Japan?|yen;JPY
E|fin|What is the official currency of Germany today?|euro;EUR
E|fin|In which city is the European Central Bank headquartered?|Frankfurt
E|fin|What is the central bank of the United States commonly called?|Federal Reserve;Fed
E|fin|What is the currency of Switzerland?|Swiss franc;franc;CHF
E|fin|In which city is the New York Stock Exchange located?|New York
E|fin|What does the abbreviation ATM stand for?|automated teller machine;automatic teller machine
E|fin|What is the currency of the United States?|dollar;USD
E|fin|What is the name of the United Kingdom's central bank?|Bank of England
E|fin|What does IPO stand for in finance?|initial public offering
E|fin|What is the currency of China?|renminbi;yuan;CNY
E|fin|What is the currency of India?|rupee;INR
E|fin|In which country is the stock index DAX based?|Germany
E|fin|What does GDP stand for?|gross domestic product
E|fin|What term describes a market in which prices are generally rising?|bull market;bull;bullish
E|fin|What does VAT stand for in taxation?|value added tax
E|fin|What is the currency of Russia?|ruble;rouble;RUB
E|fin|What does ECB stand for in European finance?|European Central Bank
M|gen|What is the capital of Kazakhstan?|Astana;Nur-Sultan
M|gen|In which year did the Berlin Wall fall?|1989
M|gen|Who wrote 'One Hundred Years of Solitude'?|Gabriel Garcia Marquez;Garcia Marquez;Marquez
M|gen|What is the chemical symbol for tungsten?|W
M|gen|Which element has the atomic number 79?|gold
M|gen|In which year did the Titanic sink?|1912
M|gen|What is the capital of Mongolia?|Ulaanbaatar;Ulan Bator
M|gen|Who painted 'The Starry Night'?|Vincent van Gogh;van Gogh
M|gen|What is the longest bone in the human body?|femur
M|gen|In which year did World War I begin?|1914
M|gen|Who formulated the laws of planetary motion published in the early 17th century?|Johannes Kepler;Kepler
M|gen|What is the capital of New Zealand?|Wellington
M|gen|What is the capital of South Korea?|Seoul
M|gen|Which physicist formulated the uncertainty principle?|Werner Heisenberg;Heisenberg
M|gen|Who wrote 'Pride and Prejudice'?|Jane Austen;Austen
M|gen|What is the smallest country in the world by area?|Vatican City;Vatican;Holy See
M|gen|What is the capital of Argentina?|Buenos Aires
M|gen|Who was the first President of the United States?|George Washington;Washington
M|gen|In which year did the French Revolution begin?|1789
M|gen|What is the largest desert in the world, excluding polar deserts?|Sahara
M|gen|What is the capital of Vietnam?|Hanoi
M|gen|Who composed the opera 'The Magic Flute'?|Wolfgang Amadeus Mozart;Mozart
M|gen|What is the most abundant element in the universe?|hydrogen
M|gen|What is the capital of Nigeria?|Abuja
M|gen|Which philosopher wrote 'Critique of Pure Reason'?|Immanuel Kant;Kant
M|gen|What is the SI unit of electrical resistance?|ohm
M|gen|In which year did humans first land on the Moon?|1969
M|gen|What is the capital of Pakistan?|Islamabad
M|gen|Who wrote the novel '1984'?|George Orwell;Orwell
M|gen|What is the capital of Switzerland?|Bern;Berne
M|gen|Which planet has the shortest day in the solar system?|Jupiter
M|gen|What is the largest moon of Saturn?|Titan
M|gen|What is the capital of Morocco?|Rabat
M|gen|Who discovered the electron in 1897?|J J Thomson;Joseph John Thomson;Thomson
M|gen|What is the capital of Peru?|Lima
M|gen|What is the longest river in Europe?|Volga
M|gen|Who wrote 'The Brothers Karamazov'?|Fyodor Dostoevsky;Dostoevsky;Dostoyevsky
M|gen|What is the capital of Chile?|Santiago
M|gen|In which year did the Soviet Union dissolve?|1991
M|gen|What is the chemical symbol for potassium?|K
M|gen|Whose 'Last Theorem' was proved by Andrew Wiles in the 1990s?|Pierre de Fermat;Fermat
M|gen|What is the capital of Ukraine?|Kyiv;Kiev
M|gen|Who painted the ceiling of the Sistine Chapel?|Michelangelo
M|gen|What is the largest internal organ in the human body?|liver
M|gen|What is the capital of Thailand?|Bangkok
M|gen|Who wrote 'Faust'?|Johann Wolfgang von Goethe;Goethe
M|gen|What is the capital of Hungary?|Budapest
M|gen|What is the name of the deepest known point in the world's oceans?|Challenger Deep;Mariana Trench
M|gen|Who proposed the theory of evolution by natural selection together with Alfred Russel Wallace?|Charles Darwin;Darwin
M|gen|What is the atomic number of carbon?|6;six
M|gen|What is the capital of Ireland?|Dublin
M|gen|What is the capital of Finland?|Helsinki
M|gen|Who was the first Chancellor of the Federal Republic of Germany?|Konrad Adenauer;Adenauer
M|gen|In which year did Germany reunify?|1990
M|gen|What is the capital of Colombia?|Bogota
M|gen|Who discovered radium together with Pierre Curie?|Marie Curie;Curie
M|gen|What is the capital of Saudi Arabia?|Riyadh
M|gen|What is the capital of Iran?|Tehran;Teheran
M|gen|Who wrote 'The Divine Comedy'?|Dante Alighieri;Dante
M|gen|What is the largest coral reef system in the world?|Great Barrier Reef
M|gen|What is the capital of Kenya?|Nairobi
M|gen|How many bones are in the adult human body?|206
M|gen|What is the capital of the Czech Republic?|Prague
M|gen|Which composer wrote 'The Four Seasons'?|Antonio Vivaldi;Vivaldi
M|gen|What is the capital of Denmark?|Copenhagen
M|gen|Which element has the chemical symbol Pb?|lead
M|gen|Who wrote 'War and Peace'?|Leo Tolstoy;Tolstoy
M|gen|What is the capital of Indonesia?|Jakarta;Nusantara
M|gen|In which year did Christopher Columbus first reach the Americas?|1492
M|gen|What is the capital of Venezuela?|Caracas
M|gen|Who invented the World Wide Web?|Tim Berners-Lee;Berners-Lee
M|gen|What is the capital of Cuba?|Havana
M|gen|What is the capital of the Philippines?|Manila
M|gen|Which physicist published the wave equation for quantum mechanics in 1926?|Erwin Schrodinger;Schrodinger;Schroedinger
M|gen|Who wrote 'Don Quixote'?|Miguel de Cervantes;Cervantes
M|gen|What is the capital of Ethiopia?|Addis Ababa
M|gen|What is the largest lake in Africa by area?|Lake Victoria;Victoria
M|gen|What is the capital of Belgium?|Brussels
M|gen|Which country has the most natural lakes?|Canada
M|gen|Who painted 'Guernica'?|Pablo Picasso;Picasso
M|gen|What is the capital of Iraq?|Baghdad
M|gen|In which year did the Chernobyl nuclear disaster occur?|1986
M|gen|Who wrote 'On the Origin of Species'?|Charles Darwin;Darwin
M|gen|What is the capital of Afghanistan?|Kabul
M|gen|Who was the British Prime Minister for most of World War II?|Winston Churchill;Churchill
M|gen|What is the name of the star closest to the Sun?|Proxima Centauri
M|gen|In which year was the Magna Carta sealed?|1215
M|gen|What is the capital of Bangladesh?|Dhaka
M|gen|Which mathematician introduced the notation e for the base of the natural logarithm?|Leonhard Euler;Euler
M|gen|Which blood type is known as the universal donor?|O negative;O neg
M|gen|What is the capital of Romania?|Bucharest
M|gen|What is the capital of Croatia?|Zagreb
M|gen|Who directed the 1960 film 'Psycho'?|Alfred Hitchcock;Hitchcock
M|gen|Who wrote the novel 'The Trial' (Der Process)?|Franz Kafka;Kafka
M|gen|What is the chemical symbol for mercury?|Hg
M|gen|In which year did the Battle of Hastings take place?|1066
M|gen|Who proved Fermat's Last Theorem?|Andrew Wiles;Wiles
M|gen|What is the highest mountain in Africa?|Kilimanjaro
M|gen|What is the longest river in Asia?|Yangtze;Chang Jiang
M|gen|What is the deepest lake in the world?|Lake Baikal;Baikal
M|gen|Which strait separates Europe from Africa at Spain and Morocco?|Strait of Gibraltar;Gibraltar
M|gen|Who was the last Tsar of Russia?|Nicholas II
M|gen|In which year did the Battle of Waterloo take place?|1815
M|gen|Which hormone is produced by the beta cells of the pancreas?|insulin
M|gen|Which city hosted the 1936 Summer Olympics?|Berlin
M|gen|In which country is the ancient city of Petra?|Jordan
M|gen|In which country is Angkor Wat?|Cambodia
M|gen|Who composed the 'Goldberg Variations'?|Johann Sebastian Bach;Bach
M|gen|Which Danish physicist proposed a model of the atom in 1913?|Niels Bohr;Bohr
M|gen|What was the name of the first artificial satellite?|Sputnik
M|fin|In which year were euro banknotes and coins introduced?|2002
M|fin|Who was the first President of the European Central Bank?|Wim Duisenberg;Duisenberg
M|fin|In which year did Lehman Brothers file for bankruptcy?|2008
M|fin|In which year was the US Federal Reserve System created?|1913
M|fin|In which year was the Bank of England founded?|1694
M|fin|What does the acronym SEPA stand for?|Single Euro Payments Area;Single Euro Payment Area
M|fin|In which Swiss city is the Bank for International Settlements headquartered?|Basel
M|fin|What is the maximum length of an IBAN in characters?|34
M|fin|How many characters long is a German IBAN?|22
M|fin|In which year was the Bretton Woods conference held?|1944
M|fin|In which US state is Bretton Woods located?|New Hampshire
M|fin|Which US banking act of 1933 separated commercial and investment banking?|Glass-Steagall;Banking Act of 1933
M|fin|In which year was the Dodd-Frank Act signed into law?|2010
M|fin|What is the minimum Liquidity Coverage Ratio required under Basel III, in percent?|100
M|fin|What is the minimum Common Equity Tier 1 capital ratio under Basel III, excluding buffers, in percent?|4.5
M|fin|What is the size of the Basel III capital conservation buffer, in percent of risk-weighted assets?|2.5
M|fin|What does SWIFT stand for in international payments?|Society for Worldwide Interbank Financial Telecommunication
M|fin|What does LIBOR stand for?|London Interbank Offered Rate;London Inter-bank Offered Rate
M|fin|Which euro overnight rate published by the ECB since 2019 replaced EONIA?|euro short-term rate;€STR;ESTR;ESTER
M|fin|What is the common short name of Germany's Federal Financial Supervisory Authority?|BaFin
M|fin|In which city is the Deutsche Bundesbank headquartered?|Frankfurt
M|fin|In which year was the Deutsche Bundesbank founded?|1957
M|fin|Which ISO standard defines the XML-based financial messaging used for SEPA payments?|ISO 20022;20022
M|fin|Which ISO standard defines three-letter currency codes such as EUR and USD?|ISO 4217;4217
M|fin|In which country were the first government-issued paper banknotes used, during the Song dynasty?|China
M|fin|How many companies are in the German DAX index since September 2021?|40;forty
M|fin|What is the highest credit rating on the S&P scale?|AAA
M|fin|Who wrote 'The Wealth of Nations'?|Adam Smith
M|fin|Which economist wrote 'The General Theory of Employment, Interest and Money'?|John Maynard Keynes;Keynes
M|fin|What is the abbreviation of the revised EU Payment Services Directive that introduced open-banking access rules?|PSD2;PSD 2
M|fin|What does NPL stand for in banking?|non-performing loan;nonperforming loan
M|fin|What is the currency of Vietnam?|dong;VND
M|fin|What is the currency of Malaysia?|ringgit;MYR
M|fin|In which year did Iceland's three largest banks collapse?|2008
M|fin|Who succeeded Mario Draghi as President of the ECB?|Christine Lagarde;Lagarde
M|fin|What does CDS stand for in finance?|credit default swap
M|fin|Which option Greek measures sensitivity to the price of the underlying?|delta
M|fin|Which option Greek measures sensitivity to volatility?|vega
M|fin|Which option Greek measures the rate of change of delta with respect to the underlying price?|gamma
M|fin|What financial ratio is net income divided by shareholders' equity?|return on equity;ROE
M|fin|What is the ISO 4217 code for the Swiss franc?|CHF
M|fin|What is the ISO 4217 code for the Japanese yen?|JPY
M|fin|What does RWA stand for in banking regulation?|risk-weighted assets
M|fin|What does EBA stand for in EU financial regulation?|European Banking Authority
H|gen|What is the capital of Burkina Faso?|Ouagadougou
H|gen|What is the capital of Bhutan?|Thimphu
H|gen|What is the capital of Eritrea?|Asmara
H|gen|What is the capital of Kyrgyzstan?|Bishkek
H|gen|What is the capital of Vanuatu?|Port Vila
H|gen|What is the capital of Suriname?|Paramaribo
H|gen|What is the capital of Tajikistan?|Dushanbe
H|gen|What is the capital of Liechtenstein?|Vaduz
H|gen|What is the capital of Malawi?|Lilongwe
H|gen|What is the capital of Palau?|Ngerulmud
H|gen|What is the capital of Tuvalu?|Funafuti
H|gen|What is the capital of the Comoros?|Moroni
H|gen|What is the capital of Lesotho?|Maseru
H|gen|What is the capital of Turkmenistan?|Ashgabat;Ashkhabad
H|gen|What is the capital of Moldova?|Chisinau
H|gen|What is the capital of Brunei?|Bandar Seri Begawan
H|gen|What is the capital of Mauritania?|Nouakchott
H|gen|What is the capital of Gabon?|Libreville
H|gen|What is the chemical symbol for antimony?|Sb
H|gen|Which element has the atomic number 74?|tungsten;wolfram
H|gen|What is the atomic number of uranium?|92
H|gen|Which element has the atomic number 50?|tin
H|gen|Which element has the atomic number 36?|krypton
H|gen|What is the atomic number of silver?|47
H|gen|In which year was the Peace of Westphalia signed?|1648
H|gen|In which year did the Congress of Vienna conclude?|1815
H|gen|Who wrote the novel 'The Master and Margarita'?|Mikhail Bulgakov;Bulgakov
H|gen|Who wrote 'The Magic Mountain' (Der Zauberberg)?|Thomas Mann
H|gen|Who wrote the novel 'Buddenbrooks'?|Thomas Mann
H|gen|Who composed the opera 'Wozzeck'?|Alban Berg
H|gen|Who composed 'The Rite of Spring'?|Igor Stravinsky;Stravinsky
H|gen|Who wrote the novel 'Pedro Paramo'?|Juan Rulfo;Rulfo
H|gen|Who wrote 'The Tin Drum' (Die Blechtrommel)?|Gunter Grass;Guenter Grass;Grass
H|gen|Who painted 'The Garden of Earthly Delights'?|Hieronymus Bosch;Bosch
H|gen|Who painted 'The Ambassadors' (1533)?|Hans Holbein;Holbein
H|gen|Who sculpted 'The Burghers of Calais'?|Auguste Rodin;Rodin
H|gen|Which mathematician published the incompleteness theorems in 1931?|Kurt Godel;Godel;Goedel
H|gen|Who proved the Poincare conjecture?|Grigori Perelman;Grigory Perelman;Perelman
H|gen|Who proved the prime number theorem in 1896 independently of Hadamard?|de la Vallee Poussin;Vallee Poussin
H|gen|Who formulated the exclusion principle in quantum mechanics?|Wolfgang Pauli;Pauli
H|gen|Which physicist predicted the positron through his relativistic equation for the electron?|Paul Dirac;Dirac
H|gen|Who discovered the neutron in 1932?|James Chadwick;Chadwick
H|gen|Who first isolated elemental fluorine in 1886?|Henri Moissan;Moissan
H|gen|Who received the first Fields Medal in 1936 together with Jesse Douglas?|Lars Ahlfors;Ahlfors
H|gen|In which year did Alexander Grothendieck receive the Fields Medal?|1966
H|gen|Which mathematician proved the Riemann hypothesis part of the Weil conjectures in 1974?|Pierre Deligne;Deligne
H|gen|What is the largest moon of Neptune?|Triton
H|gen|What is the name of the larger of Mars' two moons?|Phobos
H|gen|What is the second-largest moon of Jupiter?|Callisto
H|gen|What is the highest mountain in South America?|Aconcagua
H|gen|What is the highest mountain in Germany?|Zugspitze
H|gen|Which river flows through Baghdad?|Tigris
H|gen|Which river flows through Bratislava?|Danube;Donau
H|gen|What is the official language of Suriname?|Dutch
H|gen|What is the largest island in the Mediterranean Sea?|Sicily
H|gen|In which year did the Tunguska event occur?|1908
H|gen|Which Roman emperor issued the Edict of Milan together with Licinius in 313?|Constantine
H|gen|Who was the first emperor of the Qin dynasty?|Qin Shi Huang;Qin Shi Huangdi;Ying Zheng;Shi Huangdi
H|gen|Who wrote 'Leviathan' (1651)?|Thomas Hobbes;Hobbes
H|gen|Who wrote 'Tractatus Logico-Philosophicus'?|Ludwig Wittgenstein;Wittgenstein
H|gen|Who wrote 'Being and Time' (Sein und Zeit)?|Martin Heidegger;Heidegger
H|gen|Who wrote 'The Phenomenology of Spirit'?|Hegel
H|gen|What is the smallest bone in the human body?|stapes;stirrup
H|gen|What is the SI unit of magnetic flux?|weber
H|gen|What is the SI unit of inductance?|henry
H|gen|What is the SI unit of luminous intensity?|candela
H|gen|What is the SI unit of catalytic activity?|katal
H|gen|In which year were the first Nobel Prizes awarded?|1901
H|gen|Who won the first Nobel Prize in Physics?|Wilhelm Rontgen;Roentgen;Rontgen
H|gen|Who won the Nobel Prize in Literature in 1912?|Gerhart Hauptmann;Hauptmann
H|gen|Who won the 2016 Nobel Prize in Literature?|Bob Dylan;Dylan
H|gen|Which city hosted the 1956 Summer Olympics?|Melbourne
H|gen|Which country won the first FIFA World Cup in 1930?|Uruguay
H|gen|Who directed the 1927 film 'Metropolis'?|Fritz Lang
H|gen|Who directed the 1979 film 'Stalker'?|Andrei Tarkovsky;Tarkovsky
H|gen|What is the national animal of Scotland?|unicorn
H|gen|In which modern country is the ancient city of Persepolis?|Iran
H|gen|In which modern country was the ancient city of Carthage?|Tunisia
H|gen|In which modern country is the ancient city of Troy located?|Turkey;Turkiye
H|gen|Which German city is home to the Roman gate 'Porta Nigra'?|Trier
H|gen|Who composed the opera 'Der Freischutz'?|Carl Maria von Weber;Weber
H|gen|Who composed the opera 'Pelleas et Melisande'?|Claude Debussy;Debussy
H|gen|Which element was discovered in the solar spectrum before it was found on Earth?|helium
H|gen|Which mathematician wrote 'Disquisitiones Arithmeticae'?|Carl Friedrich Gauss;Gauss
H|gen|Who introduced the concept of schemes in algebraic geometry?|Alexander Grothendieck;Grothendieck
H|gen|Who proved in 1837 that there are infinitely many primes in arithmetic progressions?|Dirichlet
H|gen|Who proved the transcendence of pi in 1882?|Ferdinand von Lindemann;Lindemann
H|gen|Who proved the transcendence of e in 1873?|Charles Hermite;Hermite
H|gen|Who proved the four colour theorem in 1976 together with Wolfgang Haken?|Kenneth Appel;Appel
H|gen|In which year did the Space Shuttle Challenger disaster occur?|1986
H|gen|Who was the first woman to fly in space?|Valentina Tereshkova;Tereshkova
H|gen|What was the name of the lander that made the first landing on a comet in 2014?|Philae
H|gen|What was the name of the ESA mission that carried that comet lander?|Rosetta
H|gen|In which year was the Peace of Augsburg concluded?|1555
H|gen|In which year was the Edict of Nantes issued?|1598
H|gen|In which year did the Thirty Years' War begin?|1618
H|gen|Who was crowned emperor by Pope Leo III in the year 800?|Charlemagne;Charles the Great;Karl der Grosse
H|gen|Who wrote the novel 'Simplicius Simplicissimus'?|Grimmelshausen
H|gen|Who wrote the novel 'Effi Briest'?|Theodor Fontane;Fontane
H|gen|Who wrote the novel 'Berlin Alexanderplatz'?|Alfred Doblin;Doblin;Doeblin
H|gen|What is the capital of the German state of Saxony-Anhalt?|Magdeburg
H|gen|What is the capital of the German state of Mecklenburg-Western Pomerania?|Schwerin
H|gen|What is the capital of the German state of Rhineland-Palatinate?|Mainz
H|gen|What is the capital of the German state of Lower Saxony?|Hanover;Hannover
H|gen|What is the capital of the Australian state of Tasmania?|Hobart
H|gen|What is the capital of the US state of Nevada?|Carson City
H|gen|What is the capital of the US state of Vermont?|Montpelier
H|gen|What is the capital of the Canadian province of British Columbia?|Victoria
H|gen|What is the capital of the US state of South Dakota?|Pierre
H|fin|What is the currency of Bhutan?|ngultrum
H|fin|What is the currency of Kazakhstan?|tenge
H|fin|What is the currency of Georgia, the country in the Caucasus?|lari
H|fin|What is the currency of Paraguay?|guarani
H|fin|What is the currency of Angola?|kwanza
H|fin|What is the currency of Mongolia?|tugrik;togrog;tugrug
H|fin|What is the currency of Botswana?|pula
H|fin|What is the currency of Armenia?|dram
H|fin|What is the currency of Laos?|kip
H|fin|What is the currency of Guatemala?|quetzal
H|fin|What is the currency of Ethiopia?|birr
H|fin|Which Italian bank, founded in 1472, is often cited as the oldest bank still in operation?|Monte dei Paschi
H|fin|In which year was the Basel Committee on Banking Supervision established?|1974
H|fin|In which year was the first Basel Capital Accord (Basel I) published?|1988
H|fin|In which year did the ECB begin direct supervision of banks under the Single Supervisory Mechanism?|2014
H|fin|In which year did Bankhaus Herstatt fail, giving its name to settlement risk?|1974
H|fin|In which German city was Bankhaus Herstatt based?|Cologne;Koln;Koeln
H|fin|Which British bank collapsed in 1995 due to unauthorised trading by Nick Leeson?|Barings
H|fin|Which French bank revealed large losses from trader Jerome Kerviel in 2008?|Societe Generale;SocGen
H|fin|Which German lender was nationalised after the 2008 crisis involving its subsidiary Depfa?|Hypo Real Estate
H|fin|Which German payments company filed for insolvency in 2020 after a 1.9 billion euro accounting gap?|Wirecard
H|fin|Which British bank experienced a bank run in September 2007?|Northern Rock
H|fin|Which US hedge fund collapsed in 1998 and was rescued in a bailout organised by the New York Fed?|Long-Term Capital Management;LTCM
H|fin|In which year was the euro introduced as an accounting currency?|1999
H|fin|Who was President of the Deutsche Bundesbank from 1993 to 1999?|Hans Tietmeyer;Tietmeyer
H|fin|Who was the second President of the European Central Bank?|Jean-Claude Trichet;Trichet
H|fin|What is the minimum leverage ratio under Basel III, in percent?|3;three
H|fin|What is the minimum total capital ratio under Basel III excluding buffers, in percent?|8;eight
H|fin|What is the minimum Net Stable Funding Ratio under Basel III, in percent?|100
H|fin|What does MREL stand for in EU bank resolution?|minimum requirement for own funds and eligible liabilities
H|fin|What is the acronym of the EU regulation on digital operational resilience for the financial sector?|DORA
H|fin|What is the common abbreviation of Germany's Banking Act (Kreditwesengesetz)?|KWG
H|fin|In which year did the MiFID II rules start to apply?|2018
H|fin|In which year was the Bank of Japan founded?|1882
H|fin|In which year did the Swiss National Bank begin operations?|1907
H|fin|Which country's central bank is called the Riksbank?|Sweden
H|fin|Who shared the 1997 Nobel memorial prize in economics with Myron Scholes for option pricing work?|Robert Merton;Robert C Merton;Merton
H|fin|What is the name of the Eurosystem's real-time gross settlement system?|TARGET2;TARGET;T2
H|fin|What is the name of the Eurosystem securities settlement platform launched in 2015?|TARGET2-Securities;T2S
H|fin|What is the ISO 4217 currency code of the Polish zloty?|PLN
H|fin|What is the ISO 4217 currency code of the South African rand?|ZAR
H|fin|What does CRR stand for in EU banking regulation?|Capital Requirements Regulation
H|fin|What does IRB stand for in the Basel credit risk approaches?|internal ratings-based;internal rating based;internal ratings based
H|fin|What does ICAAP stand for in banking supervision?|Internal Capital Adequacy Assessment Process
H|fin|In which city has the European Banking Authority been headquartered since 2019?|Paris
"""


def load_inline():
    items = []
    for line in DATA.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        tier, dom, q, ans = [p.strip() for p in line.split("|")]
        aliases = [a.strip() for a in ans.split(";") if a.strip()]
        items.append({"id": f"q{len(items):03d}", "tier": tier, "domain": dom,
                      "question": q, "aliases": aliases})
    return items


if __name__ == "__main__":
    from collections import Counter
    it = load_inline()
    print(len(it), Counter((x["tier"], x["domain"]) for x in it))
