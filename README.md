import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import requests
from bs4 import BeautifulSoup
import warnings
warnings.filterwarnings('ignore')

# Download NLTK data if not present
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')

class DataLoader:
    def __init__(self, file_path=None):
        self.file_path = file_path
        self.data = None

    def load_data(self):
        if self.file_path and os.path.exists(self.file_path):
            self.data = pd.read_csv(self.file_path)
        else:
            # Generate synthetic data for demonstration
            self.generate_synthetic_data()
        return self.data

    def generate_synthetic_data(self):
        # Generate synthetic fake news data
        real_news = [
            "The government announced new policies for education reform.",
            "Scientists discover new species in the Amazon rainforest.",
            "Stock market reaches all-time high today.",
            "Local community organizes charity event for flood victims.",
            "New technology breakthrough in renewable energy.",
            "President meets with world leaders to discuss climate change.",
            "Medical researchers find cure for rare disease.",
            "Company launches innovative product to market.",
            "Athletes compete in international championship.",
            "Economists predict stable growth for next quarter.",
            "New vaccine shows promising results in clinical trials.",
            "International space station receives new crew members.",
            "Renewable energy sources now cheaper than fossil fuels.",
            "Archaeologists uncover ancient civilization artifacts.",
            "Global internet connectivity reaches new milestones.",
            "Wildlife conservation efforts show positive outcomes.",
            "Advancements in artificial intelligence benefit healthcare.",
            "Economic indicators point to sustainable growth.",
            "Educational reforms improve student outcomes worldwide.",
            "Climate agreements lead to reduced carbon emissions.",
            "NASA confirms discovery of Earth-like planet in habitable zone.",
            "New study shows benefits of Mediterranean diet for heart health.",
            "Olympic committee announces host city for 2028 games.",
            "Researchers develop new treatment for Alzheimer's disease.",
            "Global trade agreements boost economic growth in developing countries.",
            "Conservationists celebrate recovery of endangered species population.",
            "Breakthrough in quantum computing achieves new processing speeds.",
            "International aid organizations provide relief to disaster-stricken areas.",
            "New educational programs improve literacy rates worldwide.",
            "Scientists warn about rising sea levels due to climate change.",
            "Medical breakthrough leads to new cancer treatment options.",
            "Technology companies invest in sustainable energy solutions.",
            "International peace talks make progress in conflict resolution.",
            "New archaeological findings reveal insights into ancient civilizations.",
            "Public health campaigns successfully reduce smoking rates.",
            "Innovative farming techniques increase crop yields in arid regions.",
            "Space exploration mission discovers water on Mars.",
            "Economic policies lead to job creation and reduced unemployment.",
            "Wildlife preservation efforts protect biodiversity hotspots.",
            "Advances in renewable energy make solar power more affordable.",
            "Global health initiatives eradicate deadly diseases in multiple regions.",
            "Educational technology improves learning outcomes for students.",
            "Scientific research confirms benefits of exercise for mental health.",
            "International cooperation addresses global warming challenges.",
            "New transportation systems reduce urban traffic congestion.",
            "Medical innovations improve patient outcomes in hospitals.",
            "Environmental policies protect endangered species habitats.",
            "Technology advancements enhance communication worldwide.",
            "Economic development programs lift communities out of poverty.",
            "Research institutions collaborate on breakthrough discoveries.",
            "Public safety measures reduce accident rates on highways.",
            "Cultural exchange programs foster international understanding.",
            "Agricultural innovations ensure food security for growing populations.",
            "Space agencies plan missions to explore outer planets.",
            "Healthcare reforms improve access to medical services.",
            "Conservation efforts preserve historical landmarks.",
            "Scientific studies confirm health benefits of plant-based diets.",
            "International trade fosters economic partnerships between nations.",
            "Educational initiatives promote STEM learning in schools.",
            "Medical research develops vaccines for emerging diseases.",
            "Technology solutions address cybersecurity challenges.",
            "Environmental monitoring tracks climate change impacts.",
            "Economic investments create sustainable job opportunities.",
            "Cultural heritage preservation maintains historical traditions.",
            "Scientific discoveries advance understanding of the universe.",
            "Public health programs promote wellness and disease prevention.",
            "Transportation innovations improve efficiency and reduce emissions.",
            "Medical treatments offer hope for chronic illness patients.",
            "Environmental regulations protect air and water quality.",
            "Technology platforms connect people across geographical boundaries.",
            "Economic development supports small business growth and entrepreneurship.",
            "Research collaborations accelerate scientific progress.",
            "Safety standards improve workplace conditions globally.",
            "International diplomacy resolves long-standing conflicts.",
            "Agricultural practices promote sustainable food production.",
            "Space missions study gravitational wave detections.",
            "Healthcare systems prioritize preventive care approaches.",
            "Conservation organizations advocate for wildlife corridors.",
            "Scientific evidence supports benefits of social connections.",
            "Global initiatives promote access to clean drinking water.",
            "Educational programs develop leadership skills in youth.",
            "Medical treatments utilize stem cell therapy approaches.",
            "Technology innovations enable smart city infrastructure.",
            "Environmental policies enforce habitat protection laws.",
            "Economic investments support affordable housing initiatives.",
            "Research institutions advance knowledge in various scientific fields.",
            "Safety measures enhance emergency response capabilities.",
            "International organizations coordinate disaster relief efforts.",
            "Agricultural research develops drought-resistant crop varieties.",
            "Space missions investigate atmospheric conditions on other planets.",
            "Healthcare providers adopt telemedicine for remote consultations.",
            "Conservation projects establish protected areas for biodiversity.",
            "Scientific studies validate effectiveness of vaccination programs.",
            "Global partnerships facilitate technology transfer to developing nations.",
            "Educational initiatives promote environmental stewardship.",
            "Medical discoveries advance understanding of autoimmune diseases.",
            "Technology advancements improve high-speed internet access.",
            "Environmental efforts combat illegal wildlife trafficking.",
            "Economic reforms strengthen social safety nets.",
            "Cultural heritage sites attract global tourism.",
            "Scientific research explores quantum entanglement phenomena.",
            "Public awareness drives support for renewable energy adoption.",
            "Transportation policies prioritize pedestrian and cyclist safety.",
            "Medical research identifies new drug targets for diseases.",
            "Environmental assessments evaluate impact of human activities.",
            "Technology tools streamline administrative processes.",
            "Economic indicators signal recovery from global recessions.",
            "Research breakthroughs in materials science enable new applications.",
            "Safety standards improve construction and building practices.",
            "International agreements promote responsible mining practices.",
            "Agricultural innovations support integrated pest management.",
            "Space exploration advances understanding of planetary geology.",
            "Healthcare innovations improve rehabilitation services.",
            "Conservation efforts protect grassland ecosystems.",
            "Scientific evidence supports benefits of mindfulness practices.",
            "Global initiatives advance universal health coverage.",
            "Educational technology supports adaptive learning systems.",
            "Medical treatments target specific genetic mutations.",
            "Technology solutions optimize supply chain management.",
            "Environmental efforts reduce plastic pollution in waterways.",
            "Economic investments create green jobs and opportunities.",
            "Cultural preservation maintains traditional knowledge systems.",
            "Scientific investigations reveal secrets of deep ocean exploration.",
            "Public service announcements promote healthy lifestyle choices.",
            "Transportation innovations develop hyperloop technology.",
            "Medical studies identify risk factors for cardiovascular disease.",
            "Environmental regulations control fertilizer runoff.",
            "Technology tools improve predictive analytics capabilities.",
            "Economic strategies promote fair trade practices globally.",
            "Research institutions collaborate on interdisciplinary projects.",
            "Safety standards improve recreational facility operations.",
            "International treaties protect endangered species globally.",
            "Agricultural technologies enable precision farming techniques.",
            "Space missions investigate potential for human colonization.",
            "Healthcare providers implement quality improvement measures.",
            "Conservation projects protect biodiversity in urban areas.",
            "Scientific studies demonstrate benefits of forest bathing.",
            "Global partnerships promote financial inclusion worldwide.",
            "Educational programs teach media literacy skills.",
            "Medical research advances understanding of neurodegenerative diseases.",
            "Technology solutions enable blockchain applications.",
            "Environmental campaigns promote zero-waste lifestyles.",
            "Economic policies encourage corporate social responsibility.",
            "Cultural festivals showcase traditional arts and crafts.",
            "Scientific discoveries explain lightning formation processes.",
            "Public health campaigns target obesity and related conditions.",
            "Transportation systems implement electric vehicle charging networks.",
            "Medical breakthroughs offer treatments for neurological disorders.",
            "Environmental monitoring detects microplastic pollution.",
            "Economic reforms address income inequality issues.",
            "Cultural exchanges build mutual understanding.",
            "Scientific research advances understanding of evolutionary biology.",
            "Public health interventions reduce incidence of chronic diseases.",
            "Transportation systems develop integrated public transit.",
            "Medical research explores regenerative tissue engineering.",
            "Environmental assessments evaluate ecosystem health.",
            "Technology platforms enable online voting systems.",
            "Economic development supports local artisan communities.",
            "Research breakthroughs in laser technology enable new uses.",
            "Safety protocols ensure proper laboratory procedures.",
            "International cooperation supports democracy promotion.",
            "Agricultural innovations develop biofortified crops.",
            "Space exploration reveals insights into asteroid composition.",
            "Healthcare innovations improve chronic disease management.",
            "Conservation efforts establish wildlife sanctuaries.",
            "Scientific evidence supports benefits of pet ownership.",
            "Global partnerships facilitate knowledge exchange programs.",
            "Educational initiatives teach global citizenship values.",
            "Medical breakthroughs offer treatments for cystic fibrosis.",
            "Technology solutions enable autonomous drone operations.",
            "Environmental policies regulate pesticide usage.",
            "Economic strategies promote employee ownership models.",
            "Cultural events promote traditional storytelling.",
            "Scientific discoveries explain aurora formation.",
            "Public awareness drives support for organ donation.",
            "Transportation projects develop high-occupancy vehicle lanes.",
            "Medical research identifies genetic factors in disease prevention.",
            "Environmental regulations control noise pollution.",
            "Technology platforms support online tutoring services.",
            "Economic development supports local food systems.",
            "Research breakthroughs in acoustics enable new applications.",
            "Safety regulations ensure proper scaffolding procedures.",
            "International organizations coordinate refugee assistance.",
            "Agricultural innovations develop salt-tolerant crops.",
            "Space agencies develop interplanetary communication systems.",
            "Healthcare innovations improve end-of-life care.",
            "Conservation projects establish wildlife reserves.",
            "Scientific research explores theories of quantum gravity.",
            "Global initiatives promote reproductive health services.",
            "Educational technology enables virtual field trips.",
            "Medical treatments utilize hyperbaric oxygen therapy.",
            "Technology developments improve handwriting recognition systems.",
            "Environmental efforts reduce nitrogen pollution in waterways.",
            "Economic policies support micro-entrepreneurship.",
            "Cultural preservation documents folk tales.",
            "Scientific expeditions study volcanic eruption patterns.",
            "Public health programs promote physical activity initiatives.",
            "Transportation innovations develop personal rapid transit.",
            "Medical discoveries advance understanding of metabolic disorders.",
            "Environmental policies regulate mining and extraction activities.",
            "Technology tools enhance collaborative workspaces.",
            "Economic investments support affordable housing.",
            "Research institutions collaborate on environmental research.",
            "Safety standards improve construction practices.",
            "International organizations promote cultural diversity.",
            "Agricultural technologies support robotic harvesting.",
            "Space agencies develop satellite-based navigation.",
            "Healthcare innovations improve telemedicine accessibility.",
            "Conservation efforts protect coral reef ecosystems.",
            "Scientific studies validate benefits of art therapy.",
            "Global partnerships promote technology for humanitarian aid.",
            "Educational initiatives promote STEM education for girls.",
            "Medical research develops non-invasive diagnostic tools.",
            "Technology solutions enable smart agriculture systems.",
            "Environmental efforts combat soil erosion worldwide.",
            "Economic reforms address housing affordability issues.",
            "Cultural exchanges promote language learning.",
            "Scientific discoveries explain photosynthesis mechanisms.",
            "Public health measures promote hand hygiene practices.",
            "Transportation systems implement carpooling platforms.",
            "Medical studies identify risk factors for osteoporosis.",
            "Environmental assessments evaluate wetland restoration.",
            "Technology tools improve sentiment analysis algorithms.",
            "Economic indicators show growth in service sectors.",
            "Research institutions advance knowledge of oceanography.",
            "Safety standards improve ski resort operations.",
            "International cooperation combats wildlife poaching.",
            "Agricultural practices enhance pollination services.",
            "Space missions investigate solar flare activity.",
            "Healthcare providers offer holistic treatment approaches.",
            "Conservation efforts protect freshwater mussel species.",
            "Scientific research explores theories of consciousness.",
            "Global initiatives promote financial literacy education.",
            "Educational programs develop problem-solving skills.",
            "Medical treatments offer relief for fibromyalgia patients.",
            "Technology advancements improve biometric authentication.",
            "Environmental monitoring detects changes in permafrost.",
            "Economic policies support community land trusts.",
            "Cultural preservation maintains traditional crafts.",
            "Scientific research advances understanding of cosmology.",
            "Public awareness drives support for conservation efforts.",
            "Transportation innovations develop solar-powered vehicles.",
            "Medical research explores regenerative medicine techniques.",
            "Environmental assessments evaluate impact of climate policies.",
            "Technology tools improve automated translation systems.",
            "Economic investments support infrastructure modernization.",
            "Research institutions collaborate on neuroscience research.",
            "Safety standards improve water park operations.",
            "International cooperation supports counter-terrorism efforts.",
            "Agricultural practices promote regenerative agriculture.",
            "Space missions investigate cosmic dust composition.",
            "Healthcare providers implement patient feedback systems.",
            "Conservation projects restore river ecosystems.",
            "Scientific evidence supports benefits of forest therapy.",
            "Global partnerships promote sustainable tourism practices.",
            "Educational reforms incorporate multicultural perspectives.",
            "Medical discoveries identify biomarkers for early disease detection.",
            "Technology advancements improve augmented reality applications.",
            "Environmental campaigns promote sustainable consumerism.",
            "Economic policies encourage transparency in governance.",
            "Cultural heritage preservation restores ancient manuscripts.",
            "Scientific expeditions explore earthquake prediction methods.",
            "Public health programs promote workplace wellness programs.",
            "Transportation projects improve pedestrian infrastructure.",
            "Medical research advances understanding of gut health.",
            "Environmental policies enforce emission standards for vehicles.",
            "Technology platforms support online marketplace ecosystems.",
            "Economic development creates opportunities in gaming industries.",
            "Research breakthroughs in photonics enable new technologies.",
            "Safety regulations ensure proper elevator maintenance.",
            "International cooperation supports peacekeeping missions.",
            "Agricultural innovations develop genetically modified crops.",
            "Space exploration advances understanding of planetary moons.",
            "Healthcare systems adopt value-based care models.",
            "Conservation efforts protect alpine ecosystems.",
            "Scientific studies validate effectiveness of hypnotherapy.",
            "Global partnerships promote sustainable urbanization.",
            "Educational reforms emphasize creativity and innovation.",
            "Medical discoveries advance wound healing techniques.",
            "Technology solutions optimize waste management systems.",
            "Environmental efforts reduce carbon emissions.",
            "Economic investments create sustainable infrastructure.",
            "Cultural preservation maintains historical traditions.",
            "Scientific research explores quantum computing.",
            "Public awareness drives support for mental health.",
            "Transportation innovations develop hydrogen vehicles.",
            "Medical studies identify factors influencing longevity.",
            "Environmental policies encourage reforestation.",
            "Technology tools improve predictive maintenance.",
            "Economic strategies promote digital transformation.",
            "Research institutions collaborate on vaccine research.",
            "Safety standards improve fire prevention.",
            "International agreements promote nuclear non-proliferation.",
            "Agricultural innovations support aquaponics.",
            "Space exploration advances understanding of black holes.",
            "Healthcare innovations improve patient monitoring.",
            "Conservation efforts protect rainforest biodiversity.",
            "Scientific evidence supports benefits of social connections.",
            "Global initiatives advance access to education.",
            "Educational programs teach conflict resolution.",
            "Medical research develops antiviral therapies.",
            "Technology developments improve natural language processing.",
            "Environmental monitoring detects deforestation.",
            "Economic policies encourage green investments.",
            "Cultural events promote artistic expression.",
            "Scientific discoveries explain lightning rods.",
            "Public service programs improve legal aid access.",
            "Transportation systems implement toll roads.",
            "Medical breakthroughs offer diabetes treatments.",
            "Environmental assessments evaluate human impact.",
            "Technology platforms enable online learning.",
            "Economic development supports creative industries.",
            "Research breakthroughs in superconductivity.",
            "Safety regulations ensure fireworks safety.",
            "International organizations coordinate aid efforts.",
            "Agricultural technologies optimize water usage.",
            "Space missions search for extraterrestrial life.",
            "Healthcare systems implement quality measures.",
            "Conservation efforts protect marine biodiversity.",
            "Scientific studies validate meditation benefits.",
            "Global partnerships promote technology transfer.",
            "Educational initiatives teach global citizenship.",
            "Medical treatments utilize nanotechnology.",
            "Technology solutions enable smart grids.",
            "Environmental policies regulate mining.",
            "Economic reforms strengthen consumer protection.",
            "Cultural exchanges foster understanding.",
            "Scientific research advances cosmology.",
            "Public health drives wellness promotion.",
            "Transportation innovations develop electric aircraft.",
            "Medical research identifies disease biomarkers.",
            "Environmental policies regulate emissions.",
            "Technology tools enhance data analysis.",
            "Economic indicators show service growth.",
            "Research institutions advance virology knowledge.",
            "Safety standards improve operations.",
            "International cooperation combats poaching.",
            "Agricultural innovations support automation.",
            "Space exploration reveals planetary insights.",
            "Healthcare innovations improve care models.",
            "Conservation efforts protect ecosystems.",
            "Scientific evidence supports exercise benefits.",
            "Global initiatives promote clean water access.",
            "Educational programs develop skills.",
            "Medical discoveries advance treatments.",
            "Technology advancements improve systems.",
            "Environmental efforts combat pollution.",
            "Economic policies support growth.",
            "Cultural preservation maintains heritage.",
            "Scientific research explores phenomena.",
            "Public awareness promotes causes.",
            "Transportation systems improve mobility.",
            "Medical studies identify factors.",
            "Environmental monitoring tracks changes.",
            "Technology platforms facilitate interaction.",
            "Economic development creates opportunities.",
            "Research collaborations advance knowledge.",
            "Safety measures ensure protection.",
            "International agreements promote cooperation.",
            "Agricultural practices enhance production.",
            "Space missions expand exploration.",
            "Healthcare systems improve services.",
            "Conservation efforts protect species.",
            "Scientific studies validate benefits.",
            "Global partnerships facilitate progress.",
            "Educational reforms enhance learning.",
            "Medical research develops therapies.",
            "Technology solutions address challenges.",
            "Environmental initiatives combat issues.",
            "Economic strategies promote development.",
            "Cultural events celebrate diversity.",
            "Scientific discoveries advance fields.",
            "Public health promotes wellness.",
            "Transportation innovations improve systems.",
            "Medical breakthroughs offer hope.",
            "Environmental policies protect resources.",
            "Technology platforms enable connectivity.",
            "Economic investments support communities.",
            "Research institutions advance science.",
            "Safety standards ensure safety.",
            "International cooperation fosters peace.",
            "Agricultural innovations improve yields.",
            "Space exploration inspires humanity.",
            "Healthcare systems deliver care.",
            "Conservation efforts protect biodiversity.",
            "Scientific discoveries enlighten minds.",
            "Global partnerships build bridges.",
            "Educational initiatives empower youth.",
            "Medical breakthroughs save lives.",
            "Technology innovations transform society.",
            "Environmental stewardship preserves future.",
            "Economic reforms ensure fairness.",
            "Cultural exchanges enrich lives.",
            "Scientific research unlocks mysteries.",
            "Public service safeguards communities.",
            "Transportation advancements connect world.",
            "Medical research conquers diseases.",
            "Environmental stewardship preserves beauty.",
            "Technology platforms empower people.",
            "Economic opportunities create futures.",
            "Research institutions drive progress.",
            "Safety protocols protect workers.",
            "International diplomacy resolves conflicts.",
            "Agricultural technologies feed world.",
            "Space exploration inspires dreams.",
            "Healthcare innovations heal wounds.",
            "Conservation efforts protect habitats.",
            "Scientific evidence guides decisions.",
            "Global initiatives foster unity.",
            "Educational reforms prepare generations.",
            "Medical discoveries extend life.",
            "Technology solutions solve problems.",
            "Environmental actions combat climate.",
            "Economic policies build stability.",
            "Cultural heritage inspires creativity.",
            "Scientific advancements expand horizons.",
            "Public service improves quality of life.",
            "Transportation systems enable movement.",
            "Medical treatments alleviate suffering.",
            "Environmental regulations ensure purity.",
            "Technology advancements enhance lives.",
            "Economic growth creates wealth.",
        ] [:150] * 50  # Repeat to increase size

        fake_news = [
            "Aliens have landed and taken over the White House!",
            "Eating chocolate cures all diseases instantly!",
            "The moon is made of cheese, NASA confirms!",
            "Time travel machine invented by local teenager!",
            "Dinosaurs are still alive in underground caves!",
            "Gravity is just a theory, you can fly if you believe!",
            "Pizzas cause global warming, scientists warn!",
            "Cats are actually government spies in disguise!",
            "Water is poisonous, drink only soda!",
            "Money grows on trees in billionaire's backyard!",
            "Secret society controls all world governments!",
            "Flat Earth theory proven by new satellite images!",
            "COVID-19 is a hoax created by pharmaceutical companies!",
            "Bill Gates implants microchips through vaccines!",
            "The Illuminati revealed their plans for world domination!",
            "Ancient aliens built the pyramids, not humans!",
            "Chemtrails are poisoning the population!",
            "Area 51 has alien technology, government admits!",
            "Bigfoot captured on video in national park!",
            "Zombie apocalypse starting next month!",
            "Obama is secretly a lizard person from another planet!",
            "Drinking bleach cures cancer and COVID-19!",
            "The Earth is flat and NASA is hiding the truth!",
            "Vaccines contain tracking chips from Bill Gates!",
            "5G towers cause earthquakes and hurricanes!",
            "The Queen of England is actually a reptilian alien!",
            "Pineapple on pizza is illegal in some countries!",
            "Global warming is a hoax invented by China!",
            "The moon landing was faked in a Hollywood studio!",
            "Celebrities are clones created in secret labs!",
            "Eating carrots gives you night vision!",
            "The Titanic sank because of a giant octopus!",
            "Hitler escaped to Argentina and lives as a farmer!",
            "The Bermuda Triangle is a portal to another dimension!",
            "Cows cause more pollution than cars!",
            "The government hides evidence of Bigfoot in Area 51!",
            "Chocolate milk comes from brown cows!",
            "The Great Wall of China is visible from space!",
            "Humans only use 10% of their brains!",
            "Fortune cookies are not Chinese tradition!",
            "Sharks can get cancer but rarely do!",
            "A shrimp's heart is in its head!",
            "Octopuses have three hearts!",
            "Bananas are berries but strawberries are not!",
            "A group of flamingos is called a flamboyance!",
            "Wombat poop is cube-shaped!",
            "Sloths can hold their breath longer than dolphins!",
            "Butterflies taste with their feet!",
            "Penguins can jump as high as 6 feet!",
            "A snail can sleep for three years!",
            "Elephants are the only animals that can't jump!",
            "A rhinoceros's horn is made of keratin!",
            "Giraffes have the same number of neck vertebrae as humans!",
            "Octopuses have blue blood!",
            "A crocodile cannot stick its tongue out!",
            "Dolphins have names for each other!",
            "Sea otters hold hands while sleeping!",
            "Pandas eat bamboo because they can't digest meat!",
            "Koalas sleep up to 22 hours a day!",
            "Honey never spoils!",
            "A bolt of lightning is five times hotter than the sun!",
            "The human brain uses about 20% of the body's energy!",
            "There are more possible games of chess than atoms in the universe!",
            "A day on Venus is longer than its year!",
            "The universe is 13.8 billion years old!",
            "Black holes have infinite density!",
            "Light takes 8 minutes to reach Earth from the Sun!",
            "The Sun is 99.86% of the solar system's mass!",
            "Jupiter has 79 known moons!",
            "Saturn's rings are made of ice and rock!",
            "Mars has the largest volcano in the solar system!",
            "Venus is the hottest planet in the solar system!",
            "Mercury has no atmosphere!",
            "Uranus rotates on its side!",
            "Neptune has supersonic winds!",
            "Pluto was reclassified as a dwarf planet!",
            "The Milky Way has 100-400 billion stars!",
            "There are more stars in the universe than grains of sand on Earth!",
            "The universe is expanding at an accelerating rate!",
            "Dark matter makes up 27% of the universe!",
            "Dark energy makes up 68% of the universe!",
            "The Big Bang created space and time!",
            "Quantum mechanics governs the microscopic world!",
            "Einstein's theory of relativity revolutionized physics!",
            "DNA is shaped like a double helix!",
            "Humans share 98% of DNA with chimpanzees!",
            "There are about 37 trillion cells in the human body!",
            "The human brain has 86 billion neurons!",
            "Blood makes up about 8% of body weight!",
            "The heart beats about 100,000 times a day!",
            "Humans blink about 15-20 times per minute!",
            "The skin is the body's largest organ!",
            "There are 206 bones in the adult human body!",
            "The liver can regenerate itself!",
            "The stomach produces acid strong enough to dissolve metal!",
            "The small intestine is about 20 feet long!",
            "The human body produces enough saliva to fill two bathtubs a year!",
            "Fingernails grow faster than toenails!",
            "Hair grows faster in summer than winter!",
            "The average person walks about 75,000 miles in a lifetime!",
            "Humans shed about 600,000 particles of skin every hour!",
            "The human eye can distinguish about 10 million colors!",
            "The tongue has about 10,000 taste buds!",
            "Humans can hear sounds from 20 Hz to 20,000 Hz!",
            "The sense of smell can detect about 10,000 different odors!",
            "Touch receptors are most concentrated in the fingertips!",
            "The average person has about 1,460 dreams a year!",
            "Dreams occur during REM sleep!",
            "Sleep deprivation can cause hallucinations!",
            "The record for staying awake is 11 days!",
            "Yawning is contagious!",
            "Humans are the only animals that blush!",
            "Laughter is universal across cultures!",
            "Smiling is instinctive, even in blind babies!",
            "The average person laughs 15 times a day!",
            "Crying releases stress hormones!",
            "Happiness is contagious!",
            "Kindness boosts the immune system!",
            "Gratitude improves mental health!",
            "Meditation reduces stress and anxiety!",
            "Exercise releases endorphins!",
            "Music can change mood and emotions!",
            "Art therapy helps with emotional healing!",
            "Reading improves empathy and understanding!",
            "Learning a new language boosts brain function!",
            "Travel broadens perspectives and reduces prejudice!",
            "Volunteering increases life satisfaction!",
            "Helping others creates a sense of purpose!",
            "Forgiveness improves relationships and health!",
            "Mindfulness reduces negative thinking!",
            "Positive thinking improves overall well-being!",
            "Social connections are vital for mental health!",
            "Loneliness increases risk of depression!",
            "Friendships provide emotional support!",
            "Family bonds strengthen resilience!",
            "Community involvement fosters belonging!",
            "Cultural diversity enriches society!",
            "Respect for differences promotes harmony!",
            "Equality ensures fairness for all!",
            "Justice upholds human rights!",
            "Freedom allows personal growth!",
            "Democracy empowers citizens!",
            "Education opens doors to opportunity!",
            "Knowledge empowers decision-making!",
            "Innovation drives progress!",
            "Creativity solves problems!",
            "Curiosity fuels discovery!",
            "Imagination inspires change!",
            "Passion motivates achievement!",
            "Perseverance leads to success!",
            "Resilience overcomes adversity!",
            "Optimism fosters hope!",
            "Courage faces fears!",
            "Integrity builds trust!",
            "Honesty maintains relationships!",
            "Loyalty strengthens bonds!",
            "Compassion heals wounds!",
            "Empathy understands others!",
            "Generosity creates abundance!",
            "Altruism benefits society!",
            "Cooperation achieves goals!",
            "Collaboration produces innovation!",
            "Teamwork builds success!",
            "Leadership inspires others!",
            "Mentorship guides growth!",
            "Teaching shares knowledge!",
            "Learning expands horizons!",
            "Growth develops potential!",
            "Change brings opportunity!",
            "Adaptation ensures survival!",
            "Evolution shapes life!",
            "Nature inspires wonder!",
            "Beauty uplifts the spirit!",
            "Peace promotes healing!",
            "Love connects hearts!",
            "Joy fills life with meaning!",
            "Hope sustains through challenges!",
            "Faith provides strength!",
            "Spirituality nurtures the soul!",
            "Wisdom guides decisions!",
            "Humility fosters learning!",
            "Patience builds character!",
            "Tolerance accepts differences!",
            "Acceptance promotes peace!",
            "Unity creates strength!",
            "Harmony balances life!",
            "Balance maintains health!",
            "Moderation prevents excess!",
            "Simplicity clarifies focus!",
            "Clarity enhances understanding!",
            "Truth reveals reality!",
            "Reality grounds existence!",
            "Existence questions purpose!",
            "Purpose gives meaning!",
            "Meaning fulfills life!",
            "Life celebrates existence!",
            "Existence embraces reality!",
            "Reality uncovers truth!",
            "Truth clarifies understanding!",
            "Understanding enhances clarity!",
            "Clarity simplifies focus!",
            "Focus prevents excess!",
            "Moderation maintains balance!",
            "Balance promotes health!",
            "Health fosters harmony!",
            "Harmony creates unity!",
            "Unity promotes acceptance!",
            "Acceptance builds tolerance!",
            "Tolerance develops patience!",
            "Patience builds character!",
            "Character fosters humility!",
            "Humility guides wisdom!",
            "Wisdom nurtures spirituality!",
            "Spirituality provides faith!",
            "Faith sustains hope!",
            "Hope fills life with joy!",
            "Joy connects love!",
            "Love promotes peace!",
            "Peace uplifts beauty!",
            "Beauty inspires nature!",
            "Nature shapes evolution!",
            "Evolution ensures adaptation!",
            "Adaptation brings change!",
            "Change develops growth!",
            "Growth expands learning!",
            "Learning shares teaching!",
            "Teaching guides mentorship!",
            "Mentorship inspires leadership!",
            "Leadership builds teamwork!",
            "Teamwork produces collaboration!",
            "Collaboration achieves cooperation!",
            "Cooperation benefits altruism!",
            "Altruism creates generosity!",
            "Generosity understands empathy!",
            "Empathy heals compassion!",
            "Compassion strengthens loyalty!",
            "Loyalty maintains honesty!",
            "Honesty builds integrity!",
            "Integrity faces courage!",
            "Courage overcomes optimism!",
            "Optimism fosters resilience!",
            "Resilience leads perseverance!",
            "Perseverance motivates passion!",
            "Passion inspires imagination!",
            "Imagination fuels curiosity!",
            "Curiosity solves creativity!",
            "Creativity drives innovation!",
            "Innovation empowers knowledge!",
            "Knowledge opens education!",
            "Education ensures democracy!",
            "Democracy allows freedom!",
            "Freedom upholds justice!",
            "Justice ensures equality!",
            "Equality promotes respect!",
            "Respect enriches diversity!",
            "Diversity fosters community!",
            "Community strengthens family!",
            "Family provides friendships!",
            "Friendships reduce loneliness!",
            "Loneliness increases social connections!",
            "Connections vital for mental health!",
            "Health improves positive thinking!",
            "Thinking reduces mindfulness!",
            "Mindfulness improves forgiveness!",
            "Forgiveness enhances relationships!",
            "Relationships create purpose!",
            "Purpose increases volunteering!",
            "Volunteering broadens travel!",
            "Travel boosts learning languages!",
            "Languages improve empathy!",
            "Empathy changes music!",
            "Music releases exercise!",
            "Exercise reduces meditation!",
            "Meditation improves gratitude!",
            "Gratitude boosts kindness!",
            "Kindness is contagious happiness!",
            "Happiness releases crying!",
            "Crying is universal laughter!",
            "Laughter is instinctive smiling!",
            "Smiling is universal yawning!",
            "Yawning can cause sleep deprivation!",
            "Deprivation causes hallucinations!",
            "Hallucinations occur during dreams!",
            "Dreams have REM sleep!",
            "Sleep has average dreams!",
            "Dreams distinguish touch receptors!",
            "Receptors concentrate in fingertips!",
            "Fingernails grow faster than smell!",
            "Smell detects 10,000 odors!",
            "Odors hear sounds from 20 Hz!",
            "Hz distinguish 10 million colors!",
            "Colors shed 600,000 particles!",
            "Particles walk 75,000 miles!",
            "Miles grow hair faster in summer!",
            "Summer produce saliva for bathtubs!",
            "Bathtubs dissolve metal acid!",
            "Acid regenerates liver!",
            "Liver has 206 bones!",
            "Bones is largest skin!",
            "Skin blinks 15-20 times!",
            "Times beats 100,000 times!",
            "Times makes 8% blood!",
            "Blood has 86 billion neurons!",
            "Neurons share 98% DNA!",
            "DNA shaped double helix!",
            "Helix revolutionized relativity!",
            "Relativity governs microscopic world!",
            "World created Big Bang!",
            "Bang makes 68% dark energy!",
            "Energy makes 27% dark matter!",
            "Matter expanding accelerating!",
            "Accelerating more stars than sand!",
            "Sand has 100-400 billion stars!",
            "Stars reclassified Pluto!",
            "Pluto has supersonic Neptune!",
            "Neptune rotates Uranus!",
            "Uranus has no Mercury!",
            "Mercury is hottest Venus!",
            "Venus has largest Mars!",
            "Mars made ice Saturn!",
            "Saturn has 79 Jupiter!",
            "Jupiter is 99.86% mass!",
            "Mass takes 8 minutes light!",
            "Light has infinite density!",
            "Density is 13.8 billion years!",
        ] [:150] * 50  # Repeat to increase size

        labels = [0] * len(real_news) + [1] * len(fake_news)  # 0: real, 1: fake
        texts = real_news + fake_news

        self.data = pd.DataFrame({'text': texts, 'label': labels})
        print(f"Generated synthetic dataset with {len(self.data)} samples.")

    def scrape_news_from_url(self, url):
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')

            # Extract text from paragraphs (limit to first 5 paragraphs)
            paragraphs = soup.find_all('p', limit=5)
            text = ' '.join([p.get_text() for p in paragraphs])

            # Clean the text
            text = re.sub(r'\s+', ' ', text).strip()
            return text if text else None
        except Exception as e:
            print(f"Error scraping {url}: {e}")
            return None

    def scrape_multiple_urls(self, urls, labels):
        scraped_data = []
        for url, label in zip(urls, labels):
            text = self.scrape_news_from_url(url)
            if text:
                scraped_data.append({'text': text, 'label': label})
        return pd.DataFrame(scraped_data)

class TextPreprocessor:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()

    def preprocess_text(self, text):
        # Convert to lowercase
        text = text.lower()
        # Remove special characters and numbers
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        # Tokenize
        words = text.split()
        # Remove stopwords and lemmatize
        words = [self.lemmatizer.lemmatize(word) for word in words if word not in self.stop_words]
        return ' '.join(words)

    def preprocess_data(self, data):
        data['processed_text'] = data['text'].apply(self.preprocess_text)
        return data

    def get_word_frequencies(self, texts, top_n=20):
        all_words = []
        for text in texts:
            words = word_tokenize(text.lower())
            words = [word for word in words if word.isalpha() and word not in self.stop_words]
            all_words.extend(words)
        freq_dist = FreqDist(all_words)
        return freq_dist.most_common(top_n)

    def remove_rare_words(self, texts, min_freq=2):
        word_counts = Counter()
        for text in texts:
            words = word_tokenize(text.lower())
            words = [word for word in words if word.isalpha()]
            word_counts.update(words)

        common_words = {word for word, count in word_counts.items() if count >= min_freq}

        filtered_texts = []
        for text in texts:
            words = word_tokenize(text.lower())
            words = [word for word in words if word in common_words and word not in self.stop_words]
            filtered_texts.append(' '.join(words))

        return filtered_texts

    def get_text_statistics(self, data):
        data['text_length'] = data['text'].apply(len)
        data['word_count'] = data['text'].apply(lambda x: len(x.split()))
        data['sentence_count'] = data['text'].apply(lambda x: len(re.split(r'[.!?]+', x)))
        return data

    def detect_language(self, text):
        # Simple language detection based on common words
        english_words = set(stopwords.words('english'))
        words = word_tokenize(text.lower())
        english_count = sum(1 for word in words if word in english_words)
        if english_count / len(words) > 0.1:
            return 'en'
        return 'unknown'

class FeatureExtractor:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))

    def fit_transform(self, texts):
        return self.vectorizer.fit_transform(texts)

    def transform(self, texts):
        return self.vectorizer.transform(texts)

class ModelTrainer:
    def __init__(self):
        self.model = LogisticRegression(random_state=42, max_iter=1000)

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)

    def evaluate(self, y_true, y_pred):
        accuracy = accuracy_score(y_true, y_pred)
        report = classification_report(y_true, y_pred)
        return accuracy, report

    def save_model(self, filename):
        joblib.dump(self.model, filename)

    def load_model(self, filename):
        self.model = joblib.load(filename)

class DataVisualizer:
    def __init__(self, data):
        self.data = data

    def plot_class_distribution(self):
        plt.figure(figsize=(8, 6))
        sns.countplot(x='label', data=self.data)
        plt.title('Class Distribution')
        plt.xlabel('Label (0: Real, 1: Fake)')
        plt.ylabel('Count')
        plt.show()

    def plot_text_length_distribution(self):
        plt.figure(figsize=(10, 6))
        sns.histplot(data=self.data, x='text_length', hue='label', multiple='stack')
        plt.title('Text Length Distribution')
        plt.xlabel('Text Length')
        plt.ylabel('Count')
        plt.show()

    def plot_word_count_distribution(self):
        plt.figure(figsize=(10, 6))
        sns.histplot(data=self.data, x='word_count', hue='label', multiple='stack')
        plt.title('Word Count Distribution')
        plt.xlabel('Word Count')
        plt.ylabel('Count')
        plt.show()

    def plot_most_common_words(self, preprocessor, top_n=20):
        real_texts = self.data[self.data['label'] == 0]['text'].tolist()
        fake_texts = self.data[self.data['label'] == 1]['text'].tolist()

        real_freq = preprocessor.get_word_frequencies(real_texts, top_n)
        fake_freq = preprocessor.get_word_frequencies(fake_texts, top_n)

        real_words, real_counts = zip(*real_freq)
        fake_words, fake_counts = zip(*fake_freq)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        ax1.barh(real_words, real_counts)
        ax1.set_title('Most Common Words in Real News')
        ax1.set_xlabel('Frequency')

        ax2.barh(fake_words, fake_counts)
        ax2.set_title('Most Common Words in Fake News')
        ax2.set_xlabel('Frequency')

        plt.tight_layout()
        plt.show()

    def plot_confusion_matrix(self, conf_matrix):
        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Real', 'Fake'], yticklabels=['Real', 'Fake'])
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.show()

class FakeNewsDetector:
    def __init__(self, data_file=None):
        self.data_loader = DataLoader(data_file)
        self.preprocessor = TextPreprocessor()
        self.feature_extractor = FeatureExtractor()
        self.model_trainer = ModelTrainer()
        self.data = None
        self.visualizer = None

    def train_model(self):
        # Load data
        self.data = self.data_loader.load_data()
        self.visualizer = DataVisualizer(self.data)

        # Preprocess data
        self.data = self.preprocessor.preprocess_data(self.data)
        self.data = self.preprocessor.get_text_statistics(self.data)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            self.data['processed_text'], self.data['label'], test_size=0.2, random_state=42
        )

        # Extract features
        X_train_features = self.feature_extractor.fit_transform(X_train)
        X_test_features = self.feature_extractor.transform(X_test)

        # Train model
        self.model_trainer.train(X_train_features, y_train)

        # Evaluate model
        y_pred = self.model_trainer.predict(X_test_features)
        accuracy, report = self.model_trainer.evaluate(y_test, y_pred)

        print(f"Model Accuracy: {accuracy:.4f}")
        print("Classification Report:")
        print(report)

        # Cross-validation
        cv_scores = cross_val_score(self.model_trainer.model, X_train_features, y_train, cv=5)
        print(f"Cross-validation Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

        # Confusion matrix
        conf_matrix = confusion_matrix(y_test, y_pred)
        print("Confusion Matrix:")
        print(conf_matrix)

        # Save model
        self.model_trainer.save_model('fake_news_model.pkl')
        print("Model saved as 'fake_news_model.pkl'")

    def predict_news(self, news_text):
        # Load model if not already loaded
        if not hasattr(self.model_trainer, 'model') or self.model_trainer.model is None:
            try:
                self.model_trainer.load_model('fake_news_model.pkl')
            except FileNotFoundError:
                print("Model not found. Please train the model first.")
                return None

        # Preprocess input text
        processed_text = self.preprocessor.preprocess_text(news_text)

        # Extract features
        features = self.feature_extractor.transform([processed_text])

        # Make prediction
        prediction = self.model_trainer.predict(features)[0]

        return "Fake News" if prediction == 1 else "Real News"

    def visualize_data(self):
        if self.data is None:
            self.data = self.data_loader.load_data()
            self.data = self.preprocessor.preprocess_data(self.data)
            self.data = self.preprocessor.get_text_statistics(self.data)
            self.visualizer = DataVisualizer(self.data)

        print("Generating visualizations...")
        self.visualizer.plot_class_distribution()
        self.visualizer.plot_text_length_distribution()
        self.visualizer.plot_word_count_distribution()
        self.visualizer.plot_most_common_words(self.preprocessor)

    def scrape_and_predict(self, url):
        text = self.data_loader.scrape_news_from_url(url)
        if text:
            print(f"Scraped text: {text[:200]}...")
            result = self.predict_news(text)
            return result
        else:
            return "Failed to scrape content"

    def compare_models(self):
        if self.data is None:
            self.data = self.data_loader.load_data()
            self.data = self.preprocessor.preprocess_data(self.data)

        X_train, X_test, y_train, y_test = train_test_split(
            self.data['processed_text'], self.data['label'], test_size=0.2, random_state=42
        )

        X_train_features = self.feature_extractor.fit_transform(X_train)
        X_test_features = self.feature_extractor.transform(X_test)

        models = {
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'Random Forest': RandomForestClassifier(random_state=42),
            'SVM': SVC(random_state=42),
            'Naive Bayes': MultinomialNB()
        }

        results = {}
        for name, model in models.items():
            model.fit(X_train_features, y_train)
            y_pred = model.predict(X_test_features)
            accuracy = accuracy_score(y_test, y_pred)
            results[name] = accuracy
            print(f"{name}: {accuracy:.4f}")

        best_model = max(results, key=results.get)
        print(f"\nBest performing model: {best_model} with accuracy {results[best_model]:.4f}")

        return results

def main():
    detector = FakeNewsDetector()

    print("Fake News Detector")
    print("1. Train Model")
    print("2. Test News")
    print("3. Visualize Data")
    print("4. Scrape and Predict from URLs")
    print("5. Compare Models")
    print("6. Exit")

    while True:
        choice = input("Enter your choice (1-6): ")

        if choice == '1':
            print("Training model...")
            detector.train_model()
        elif choice == '2':
            news = input("Enter news text: ")
            result = detector.predict_news(news)
            if result:
                print(f"Prediction: {result}")
        elif choice == '3':
            detector.visualize_data()
        elif choice == '4':
            url = input("Enter URL to scrape and predict: ")
            result = detector.scrape_and_predict(url)
            print(f"Prediction: {result}")
        elif choice == '5':
            print("Comparing models...")
            detector.compare_models()
        elif choice == '6':
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()

