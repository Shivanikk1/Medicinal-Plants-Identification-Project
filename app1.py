# Importing essential libraries and modules

from flask import Flask, flash, render_template, request, Markup,redirect
import numpy as np
import pandas as pd

import requests
import config
import pickle
import io
from PIL import Image
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

# ==============================================================================================

# -------------------------LOADING THE TRAINED MODELS -----------------------------------------------

# Loading plant disease classification model

plant_data=[

{
   "plant_name": "Aloevera",
    "medicine_content": "Aloin, polysaccharides, vitamins, minerals",
    "diseases_cured": ["Skin conditions", "diabetes","obesity","Digestive problems"],
    "age_restrictions": " age restrictions",
    "gender_restrictions": "No gender restrictions",
    "pregnant_use_restriction": "Alovera should not be taken orally during pregnancy as aloe latex contain anthraquinones that may stimulate utreus and initiate premature labor or possibly cause abortion",
    "mode_of_use": "Topical application or oral consumption",
    "doses_for_day": "2-3 times a day"
},
{
    "plant_name": "Amla",
    "medicine_content": "Vitamin C, tannins, flavonoids, carotene",
    "diseases_cured": ["Boosts immunity", "Improves digestion","Respiratory disorders", "Hair health"],
    "age_restrictions": "Safe for all ages",
    "gender_restrictions": "No gender restrictions",
    "pregnant_use_restriction": "Generally safe during pregnancy, amla is rich in various nutrients, antioxidants, fibers and vitamin C",
    "mode_of_use": "Consumed raw, as a juice, or in various recipes",
    "doses_for_day": "1-2 amla fruits or 20-30 ml of amla juice per day"
},
 {
    "plant_name": "Amruthaballi",
    "medicine_content": "Alkaloids, flavonoids, glycosides",
    "diseases_cured": ["Boosts immunity", "Diabetes","Kidney infections", "Respiratory disorders"],
    "age_restrictions": "Safe for all ages",
    "gender_restrictions": "No gender restrictions",
    "pregnant_use_restriction": "Avoid during pregnancy, Consult a doctor before use",
    "mode_of_use": "Decoction, powder, or as directed by an Ayurvedic practitioner",
    "doses_for_day": "Typically 1-2 teaspoons of powder or as per practitioner's advice"
},
{
    "plant_name": "Arali",
    "medicine_content": "Glycosides, polysaccharides,triterpenes, tannins",
    "diseases_cured": ["Skin conditions", "cancer","cardiotonic", "Asthma"],
    "age_restrictions": "Safe for adults; consult a doctor for children",
    "gender_restrictions": "No gender restrictions",
    "pregnant_use_restriction": "Avoid using during pregnancy, it might cause abortion or birth defects",
    "mode_of_use": "Topical application of the paste or as directed by an Ayurvedic practitioner",
    "doses_for_day": "As directed by an Ayurvedic practitioner for internal use; for external use, apply as needed"
},
{
    "plant_name": "Astma_weed",
    "medicine_content": "Terpenoids, tannin, cannabinoids, flavonoids",
    "diseases_cured": ["Asthma", "skin ailments","Inflammatory conditions"],
    "age_restrictions": "Safe for adults; consult a doctor for children",
    "gender_restrictions": "No gender restrictions",
    "pregnant_use_restriction": "Not recommended during pregnancy; consult a doctor",
    "mode_of_use": "Inhalation, infusion, or as recommended by a healthcare professional",
    "doses_for_day": "Dosage may vary; consult a healthcare professional for personalized advice"
},
{
    "plant_name": "Badipala",
    "medicine_content": "Alkaloids, saponins, flavonoids",
    "diseases_cured": ["Digestive disorders", "Liver health", "Fever"],
    "age_restrictions": "Safe for adults; consult a doctor for children",
    "gender_restrictions": "No gender restrictions",
    "pregnant_use_restriction": "Consult a doctor during pregnancy",
    "mode_of_use": "Decoction, powder, or as directed by a herbalist",
    "doses_for_day": "Typically 1-2 teaspoons of powder or as per herbalist's advice"
},
{
    "plant_name": "Balloon_Vine",
    "medicine_content": "Steroidal saponins, flavonoids, phenols",
    "diseases_cured": ["Inflammation", "Joint pain", "Skin diseases","nervous disorders"],
    "age_restrictions": "Safe for adults; consult a doctor for children",
    "gender_restrictions": "No gender restrictions",
    "pregnant_use_restriction": "Ballon Wine should not be used during pregnancy",
    "mode_of_use": "Infusion, topical application, or as advised by a herbal practitioner",
    "doses_for_day": "Dosage may vary; follow herbal practitioner's recommendations"
},
{
    "plant_name": "Bamboo",
    "medicine_content": "Sodium, potassium, calcium, iron, antioxidants",
    "diseases_cured": ["Asthma", "Cough", "Digestive issues"],
    "age_restrictions": "Generally safe for all ages",
    "gender_restrictions": "No gender restrictions",
    "pregnant_use_restriction": "Generally safe during pregnancy, But should not be eaten more 1-2 times in a month",
    "mode_of_use": "Consumed as shoots, extracts, Bamboo Powder, Bamboo decoction and tea",
    "doses_for_day": "Dosage may vary "
},
{
    "plant_name": "Beans",
    "medicine_content": "Protein, fiber, vitamins (B-complex, folate), minerals (iron, magnesium)",
    "diseases_cured": ["Heart health", "Weight management", "Diabetes prevention"],
    "age_restrictions": "Safe for all ages",
    "gender_restrictions": "No gender restrictions",
    "pregnant_use_restriction": "Generally safe during pregnancy, as they are good source of protein, iron, folate, potassium and magnesium",
    "mode_of_use": "Cooked, raw, or as part of various dishes",
    "doses_for_day": "Varies based on dietary needs"
},
{
    "plant_name": "Betel",
    "medicine_content": "Arecoline, protein, minerals, vitamin C, essential oil",
    "diseases_cured": ["Digestive issues", "Oral health", "Cancer","fungal infection"],
    "age_restrictions": "Safe for adults; It is not safe for small children",
    "gender_restrictions": "No gender restrictions",
    "pregnant_use_restriction": "Avoid during pregnancy; consult a healthcare professional",
    "mode_of_use": "Chew the leaves, acts as a mouth freshener",
    "doses_for_day": "Dosage varies; moderate consumption recommended, avoid excessive use"
},
{
    "plant_name": "Bhrami",
    "medicine_content": "Bacoside-A and B,, alkaloids, saponins",
    "diseases_cured": ["Memory enhancement", "Stress relief", "Anxiety"],
    "age_restrictions": "Safe for all ages",
    "gender_restrictions": "No gender restrictions",
    "pregnant_use_restriction": "Avoid using during pregnancy, as it can have uterine-stimulating effects",
    "mode_of_use": "Powder, capsules, or as recommended by an Ayurvedic practitioner",
    "doses_for_day": "Typically 250-500 mg per day, or as per practitioner's advice"
},
{
    "plant_name": "Bringaraja",
    "medicine_content": "Alkaloids, flavonoids, ploypeptides, steroids",
    "diseases_cured": ["Hair health", "Liver disorders", "Anemia"],
    "age_restrictions": "Safe for adults; consult a doctor for children",
    "gender_restrictions": "No gender restrictions",
    "pregnant_use_restriction": "Avoid bhringaraja during pregenancy",
    "mode_of_use": "Hair oil, powder, or as directed by an Ayurvedic practitioner",
    "doses_for_day": "Dosage varies; follow the guidance of an Ayurvedic practitioner"
},
{
    "plant_name": "Caricature",
    "medicine_content": "Flavonoids, alkaloids, tannins",
    "diseases_cured": ["Anti-inflammatory", "Antioxidant properties"],
    "age_restrictions": "Safe for adults; consult a botanist or herbalist for children",
    "gender_restrictions": "No gender restrictions",
    "pregnant_use_restriction": "Consult a botanist or healthcare professional during pregnancy",
    "mode_of_use": "Leaves used in traditional medicine, typically as infusions or extracts",
    "doses_for_day": "Dosage may vary; follow the advice of a botanist or herbalist"
},
{
    "plant_name": "Castor",
    "medicine_content": "Ricinoleic acid, triglycerides, antioxidants",
    "diseases_cured": ["Laxative properties", "Skin conditions", "Joint pain"],
    "age_restrictions": "Safe for adults; consult a doctor for children",
    "gender_restrictions": "No gender restrictions",
    "pregnant_use_restriction": "consult a healthcare professional during pregnancy",
    "mode_of_use": "Oil extracted from seeds, applied topically or taken orally as a laxative",
    "doses_for_day": "Dosage varies; consult with a healthcare professional for personalized recommendations"
},
{
    "plant_name": "Catharanthus",
    "medicine_content": "Alkaloids, vinblastine, vincristine",
    "diseases_cured": ["cancer", "Blood pressure regulation","kidney and liver diseases"],
    "age_restrictions": "Consult a healthcare professional for appropriate use for adults and children",
    "gender_restrictions": "No gender restrictions",
    "pregnant_use_restriction": "Avoid during pregnancy; consult a healthcare professional",
    "mode_of_use": "Extracts from the plant, often in the form of medications",
    "doses_for_day": "Dosage is specific to medical conditions; follow the advice of a healthcare professional"
},
{
    "plant_name": "Chakte",
    "medicine_content": "Tannins, alkaloids, flavonoids",
    "diseases_cured": ["Anti-inflammatory", "Digestive disorders", "Skin conditions"],
    "age_restrictions": "Safe for adults; consult a doctor for children",
    "gender_restrictions": "No gender restrictions",
    "pregnant_use_restriction": "Consult a healthcare professional during pregnancy",
    "mode_of_use": "Traditionally used as a decoction, powder, or topical application",
    "doses_for_day": "Dosage may vary; follow the guidance of a traditional healer or healthcare professional"
},
{
    "plant_name": "Chilly",
    "medicine_content": "Capsaicin, vitamins (A, C), antioxidants, minerals(sodium, iron and calcium)",
    "diseases_cured": ["Pain relief", "Metabolism boost", "Immune system support"],
    "age_restrictions": "Generally safe for all ages; moderate consumption recommended for children",
    "gender_restrictions": "No gender restrictions",
    "pregnant_use_restriction": "Safe during pregnancy in moderation",
    "mode_of_use": "Cooked in various dishes, raw, or as a spice",
    "doses_for_day": "Dosage varies based on tolerance and culinary preferences; consume in moderation"
},
{
    "plant_name": "Citron lime (herelikai)",
    "medicine_content": "Vitamin C, flavonoids, essential oils, antioxidants",
    "diseases_cured": ["Boosts immunity", "Digestive health", "Blood pressure"],
    "age_restrictions": "Safe for all ages",
    "gender_restrictions": "No gender restrictions",
    "pregnant_use_restriction": "Generally safe during pregnancy, as they are excellent source of vitamin C",
    "mode_of_use": "Consumed raw, as juice, or used in culinary dishes",
    "doses_for_day": "Varies based on culinary preferences; recommended daily intake of vitamin C"
},
{
    "plant_name": "Coffee",
    "medicine_content": "Caffeine, antioxidants",
    "diseases_cured": ["Mental alertness", "Improved mood", "Antioxidant properties"],
    "age_restrictions": "Moderate consumption for adults; consult a doctor for children",
    "gender_restrictions": "No gender restrictions",
    "pregnant_use_restriction": "Moderate consumption during pregnancy; consult a healthcare professional",
    "mode_of_use": "Brewed as a beverage",
    "doses_for_day": "Dosage varies based on personal tolerance; moderate consumption recommended"
},
{
    "plant_name": "Common rue(naagdalli)",
    "medicine_content": "Alkaloids, flavonoids, essential oils",
    "diseases_cured": ["Anti-inflammatory", "Digestive disorders", "Respiratory issues"],
    "age_restrictions": "Consult a herbalist or healthcare professional for appropriate use in children",
    "gender_restrictions": "No gender restrictions",
    "pregnant_use_restriction": "Avoid during pregnancy, as it is toxic to embryo or fetus",
    "mode_of_use": "Infusion, decoction, or as directed by a herbal practitioner",
    "doses_for_day": "Dosage may vary; follow the guidance of a herbal practitioner or healthcare professional"
},
{
    "plant_name": "Coriander",
    "medicine_content": "Vitamins (C, K), antioxidants, minerals, essential oils",
    "diseases_cured": ["Digestive health", "Anti-inflammatory", "Blood sugar regulation"],
    "age_restrictions": "Safe for all ages",
    "gender_restrictions": "No gender restrictions",
    "pregnant_use_restriction": "Generally safe during pregnancy, but consult a nutritionist",
    "mode_of_use": "Leaves, seeds, or as a spice in culinary dishes",
    "doses_for_day": "Dosage varies based on culinary preferences; incorporate in daily diet"
},
{
    "plant_name": "Curry",
    "medicine_content": "Carbazole alkaloids, essential oils",
    "diseases_cured": ["Anemia", "Anti-inflammatory", "Hair health"],
    "age_restrictions": "Safe for all ages",
    "gender_restrictions": "No gender restrictions",
    "pregnant_use_restriction": "Generally safe during pregnancy",
    "mode_of_use": "Leaves used in cooking, often as a seasoning",
    "doses_for_day": "Dosage varies based on culinary preferences; commonly used in daily cooking"
},
{
    "plant_name": "Doddpathre",
    "medicine_content": "Terpenoids, flavonoids, essential oils",
    "diseases_cured": ["Heart disease", "Respiratory conditions", "Digestive issues","Stroke"],
    "age_restrictions": "Safe for adults; consult a herbalist or healthcare professional for children",
    "gender_restrictions": "No gender restrictions",
    "pregnant_use_restriction": "Consult a healthcare professional during pregnancy",
    "mode_of_use": "Leaves used in decoctions, infusions, or as directed by a herbal practitioner",
    "doses_for_day": "Dosage may vary; follow the guidance of a herbal practitioner or healthcare professional"
},
{
    "plant_name": "Drumstick",
    "medicine_content": "Vitamins (A, C, B-complex), minerals (iron, calcium), antioxidants",
    "diseases_cured": ["Acne", "Improves digestion", "Joint health","Liver diseases"],
    "age_restrictions": "Generally safe for all ages",
    "gender_restrictions": "No gender restrictions",
    "pregnant_use_restriction": "Generally not safe during pregnancy, as it acts as an anti-fertility agent",
    "mode_of_use": "Pods, leaves, or flowers used in cooking or as supplements",
    "doses_for_day": "Dosage varies based on culinary preferences; incorporate in daily diet"
},
{
    "plant_name": "Ekka",
    "medicine_content": "Cardiac glycosides, alkaloids, flavonoids",
    "diseases_cured": ["Neurological disorders", "Wound healing", "Respiratory disorders"],
    "age_restrictions": "Consult a herbalist or healthcare professional for appropriate use in children",
    "gender_restrictions": "No gender restrictions",
    "pregnant_use_restriction": "Avoid during pregnancy; consult a healthcare professional",
    "mode_of_use": "Latex and leaves used traditionally, often in poultices or decoctions",
    "doses_for_day": "Dosage may vary; follow the guidance of a herbal practitioner or healthcare professional"
},
{
    "plant_name": "Eucalyptus",
    "medicine_content": "Eucalyptol, flavonoids, tannins",
    "diseases_cured": ["Respiratory conditions", "Anti-inflammatory", "Antibacterial"],
    "age_restrictions": "Consult a healthcare professional for appropriate use in children",
    "gender_restrictions": "No gender restrictions",
    "pregnant_use_restriction": "Consult a healthcare professional during pregnancy",
    "mode_of_use": "Leaves and oil used in steam inhalation, topical applications, or as directed by a healthcare practitioner",
    "doses_for_day": "Dosage may vary; follow the advice of a healthcare professional or herbalist"
},
{
    "plant_name": "Ganigale",
    "medicine_content": "Flavonoids, phenolic compounds",
    "diseases_cured": ["Anti-inflammatory", "Antioxidant properties", "Digestive health"],
    "age_restrictions": "Generally safe for adults; consult a herbalist or healthcare professional for children",
    "gender_restrictions": "No gender restrictions",
    "pregnant_use_restriction": "Consult a healthcare professional during pregnancy",
    "mode_of_use": "Leaves and stems used in decoctions, infusions, or as directed by a herbal practitioner",
    "doses_for_day": "Dosage may vary; follow the guidance of a herbal practitioner or healthcare professional"
},
{
    "plant_name": "Ganike",
    "medicine_content": "Alkaloids, flavonoids, vitamins (A, C)",
    "diseases_cured": ["Anti-inflammatory", "Antioxidant properties", "Digestive health"],
    "age_restrictions": "Generally safe for adults; consult a herbalist or healthcare professional for children",
    "gender_restrictions": "No gender restrictions",
    "pregnant_use_restriction": "Consult a healthcare professional during pregnancy",
    "mode_of_use": "Leaves and berries used in decoctions, infusions, or as directed by a herbal practitioner",
    "doses_for_day": "Dosage may vary; follow the guidance of a herbal practitioner or healthcare professional"
},
{
    "plant_name": "Gasagase",
    "medicine_content": "Fatty acids, volatile oils, antioxidants",
    "diseases_cured": ["Digestive health", "Relief from cough", "Sleep aid"],
    "age_restrictions": "Safe for all ages; consult a healthcare professional for infants",
    "gender_restrictions": "No gender restrictions",
    "pregnant_use_restriction": "Generally safe during pregnancy, but consult a healthcare professional",
    "mode_of_use": "Seeds used in cooking, often in desserts or beverages",
    "doses_for_day": "Dosage varies based on culinary preferences; commonly used in moderate amounts"
},
{
    "plant_name": "Ginger",
    "medicine_content": "Gingerol, volatile oils, antioxidants",
    "diseases_cured": ["Anti-inflammatory", "Digestive health", "Nausea relief"],
    "age_restrictions": "Generally safe for all ages; consult a healthcare professional for infants",
    "gender_restrictions": "No gender restrictions",
    "pregnant_use_restriction": "Generally safe during pregnancy, they help in reducing nausea during pregnancy",
    "mode_of_use": "Rhizomes used in cooking, as tea, or in various remedies",
    "doses_for_day": "Dosage varies based on culinary preferences and health needs; commonly used in moderate amounts"
},
{
    "plant_name": "Globe Amaranth",
    "medicine_content": "Flavonoids, saponins, tannins",
    "diseases_cured": ["Headaches", "High blood pressure", "Digestive issues"],
    "age_restrictions": "Generally safe for all ages",
    "gender_restrictions": "No gender restrictions",
    "pregnant_use_restriction": "Consult a healthcare professional during pregnancy",
    "mode_of_use": "Flowers used in teas, infusions, or decorative purposes",
    "doses_for_day": "Dosage may vary; follow the guidance of a herbal practitioner or healthcare professional"
},
{
    "plant_name": "Guava",
    "medicine_content": "Vitamin C, dietary fiber, antioxidants",
    "diseases_cured": ["Boosts immunity", "Digestive health", "Heart health"],
    "age_restrictions": "Safe for all ages",
    "gender_restrictions": "No gender restrictions",
    "pregnant_use_restriction": "Generally safe during pregnancy, it helps in preventing anaemia",
    "mode_of_use": "Fruit consumed raw, in juices, or included in various recipes",
    "doses_for_day": "Dosage varies based on dietary preferences; commonly consumed in moderate amounts"
},
{
    "plant_name": "Henna",
    "medicine_content": "Lawson dye, tannins",
    "diseases_cured": ["Natural hair dye", "Skin conditions", "infections"],
    "age_restrictions": "Generally safe for adults, except childern",
    "gender_restrictions": "No gender restrictions",
    "pregnant_use_restriction": "Avoid during pregnancy; consult a healthcare professional",
    "mode_of_use": "Leaves used for natural dyeing of hair and skin",
    "doses_for_day": "Dosage is specific to the application; follow instructions for hair or skin use"
},
{
    "plant_name": "Hibiscus",
    "medicine_content": "Flavonoids, polyphenols, vitamins (A, C)",
    "diseases_cured": ["Lowering blood pressure", "High cholesterol", "Hair health"],
    "age_restrictions": "Generally safe for all ages",
    "gender_restrictions": "No gender restrictions",
    "pregnant_use_restriction": "Advised to limit hibiscus intake during pregnancy due to its potential to low blood pressure and its association with uterine contractions ",
    "mode_of_use": "Flowers used in teas, infusions, or applied topically for hair care",
    "doses_for_day": "Dosage varies based on application; commonly used in moderate amounts for tea or hair treatments"
},
{
    "plant_name": "Honge",
    "medicine_content": "Triterpenoids, flavonoids, fatty acids",
    "diseases_cured": ["Anti-inflammatory", "Skin conditions", "Antioxidant properties"],
    "age_restrictions": "Generally safe for all ages",
    "gender_restrictions": "No gender restrictions",
    "pregnant_use_restriction": "Consult a healthcare professional during pregnancy",
    "mode_of_use": "Seeds and oil used in traditional medicine, cooking, or applied topically",
    "doses_for_day": "Dosage may vary based on application; follow the advice of a healthcare professional or herbalist"
},
{
    "plant_name": "Insulin",
    "medicine_content": "Alkaloids, flavonoids, saponins",
    "diseases_cured": ["Blood sugar regulation", "Anti-inflammatory", "Digestive health"],
    "age_restrictions": "Consult a healthcare professional for appropriate use in children",
    "gender_restrictions": "No gender restrictions",
    "pregnant_use_restriction": "Consult a healthcare professional during pregnancy",
    "mode_of_use": "Leaves and rhizomes used in decoctions, infusions, or as directed by a herbal practitioner",
    "doses_for_day": "Dosage may vary; follow the guidance of a herbal practitioner or healthcare professional"
},
{
    "plant_name": "Jackfruit",
    "medicine_content": "Dietary fiber, vitamins (C, B6), antioxidants",
    "diseases_cured": ["Digestive health", "Immune system support", "Heart health"],
    "age_restrictions": "Safe for all ages",
    "gender_restrictions": "No gender restrictions",
    "pregnant_use_restriction": "Generally safe during pregnancy, but consult a nutritionist",
    "mode_of_use": "Fruit consumed in various culinary dishes, raw or cooked",
    "doses_for_day": "Dosage varies based on culinary preferences; commonly consumed in moderate amounts"
},
{
    "plant_name": "Jasmine",
    "medicine_content": "Essential oils, flavonoids,",
    "diseases_cured": ["Stress relief", "Liver disease", "Skin conditions"],
    "age_restrictions": "Generally safe for all ages",
    "gender_restrictions": "No gender restrictions",
    "pregnant_use_restriction": "Consult a healthcare professional during pregnancy",
    "mode_of_use": "Flowers used in aromatherapy, teas, or as extracts for skin applications",
    "doses_for_day": "Dosage varies based on application; commonly used in moderate amounts for aromatherapy or skin treatments"
},
{
    "plant_name": "Kambajala",
    "medicine_content": "Flavonoids, quercetin, kaempferol",
    "diseases_cured": ["Blood sugar regulation", "Anti-inflammatory", "Antioxidant properties"],
    "age_restrictions": "Consult a healthcare professional for appropriate use in children",
    "gender_restrictions": "No gender restrictions",
    "pregnant_use_restriction": "Consult a healthcare professional during pregnancy",
    "mode_of_use": "Leaves and bark used in infusions, decoctions, or as directed by a herbal practitioner",
    "doses_for_day": "Dosage may vary; follow the guidance of a herbal practitioner or healthcare professional"
},
{
    "plant_name": "Kasambruga",
    "medicine_content": "Resveratrol, flavonoids, polyphenols",
    "diseases_cured": ["Asthma", "Heart health", "Anti-inflammatory"],
    "age_restrictions": "Generally safe for adults; consult a healthcare professional for children",
    "gender_restrictions": "No gender restrictions",
    "pregnant_use_restriction": "Consult a healthcare professional during pregnancy",
    "mode_of_use": "Fruits and leaves used, often consumed as grapes, grape juice, or in the form of supplements",
    "doses_for_day": "Dosage varies based on form; follow recommended serving sizes for grapes, juice, or supplements"
},
{
    "plant_name": "Kohlrabi",
    "medicine_content": "Vitamins (C, B6), fiber, antioxidants",
    "diseases_cured": ["Digestive health", "Immune system support", "Bone health"],
    "age_restrictions": "Safe for all ages",
    "gender_restrictions": "No gender restrictions",
    "pregnant_use_restriction": "Generally safe during pregnancy, but consult a nutritionist",
    "mode_of_use": "Bulbous stem and leaves used in cooking, often in salads, soups, or as a side dish",
    "doses_for_day": "Dosage varies based on culinary preferences; commonly consumed in moderate amounts"
},
{
    "plant_name": "Lantana",
    "medicine_content": "Alkaloids, flavonoids, triterpenes",
    "diseases_cured": ["Malaria", "Fever", "Respiratory conditions","Influenza"],
    "age_restrictions": "Consult a herbalist or healthcare professional for appropriate use in children and adults",
    "gender_restrictions": "No gender restrictions",
    "pregnant_use_restriction": "Avoid during pregnancy, they present teratogenic alterations in their offspring",
    "mode_of_use": "Leaves and flowers used in infusions, decoctions, or as directed by a herbal practitioner",
    "doses_for_day": "Dosage may vary; follow the guidance of a herbal practitioner or healthcare professional"
},
{
    "plant_name": "Lemon",
    "medicine_content": "Vitamin C, citric acid, flavonoids, glycosides",
    "diseases_cured": ["Boosts immunity", "Digestive health", "Antioxidant properties"],
    "age_restrictions": "Safe for all ages",
    "gender_restrictions": "No gender restrictions",
    "pregnant_use_restriction": "Generally safe during pregnancy, with minimal use",
    "mode_of_use": "Fruit consumed raw, in juices, or used as a flavoring agent",
    "doses_for_day": "Dosage varies based on culinary preferences; commonly consumed in moderate amounts"
},
{
    "plant_name": "Lemongrass",
    "medicine_content": "Citronella, citral, antioxidants",
    "diseases_cured": ["Digestive health", "Relieves stress", "Anti-inflammatory"],
    "age_restrictions": "Generally safe for all ages",
    "gender_restrictions": "No gender restrictions",
    "pregnant_use_restriction": "Lemongrass is unsafe when taken by mouth during pregnancy, it can stimulate uterus and menstrual flow which causes miscarriage",
    "mode_of_use": "Stalks and leaves used in teas, infusions, or as a flavoring agent in cooking",
    "doses_for_day": "Dosage varies based on culinary preferences; commonly consumed in moderate amounts"
},
{
    "plant_name": "Malabar Nut",
    "medicine_content": "Vasaka alkaloids, vasicine, vasicinone",
    "diseases_cured": ["Respiratory conditions", "Anti-inflammatory", "Cough relief"],
    "age_restrictions": "Consult a healthcare professional for appropriate use in children",
    "gender_restrictions": "No gender restrictions",
    "pregnant_use_restriction": "Not safe during pregnancy",
    "mode_of_use": "Leaves and roots used in decoctions, infusions, or as directed by a herbal practitioner",
    "doses_for_day": "Dosage may vary; follow the guidance of a herbal practitioner or healthcare professional"
},
{
    "plant_name": "Malabar Spinach",
    "medicine_content": "Vitamins (A, C), iron, antioxidants",
    "diseases_cured": ["Eye health", "Immune system support", "Bone health"],
    "age_restrictions": "Safe for all ages",
    "gender_restrictions": "No gender restrictions",
    "pregnant_use_restriction": "Generally safe during pregnancy, but consult a nutritionist",
    "mode_of_use": "Leaves and stems used in cooking, often in salads, soups, or stir-fries",
    "doses_for_day": "Dosage varies based on culinary preferences; commonly consumed in moderate amounts"
},
{
    "plant_name": "Mango",
    "medicine_content": "Vitamins (A, C, E), beta-carotene, antioxidants",
    "diseases_cured": ["Boosts immunity", "Digestive health", "Eye health"],
    "age_restrictions": "Generally safe for all ages",
    "gender_restrictions": "No gender restrictions",
    "pregnant_use_restriction": "Generally safe during pregnancy",
    "mode_of_use": "Fruit consumed fresh, in juices, smoothies, or as part of various dishes",
    "doses_for_day": "Dosage varies based on culinary preferences; commonly consumed in moderate amounts"
},
{
    "plant_name": "Marigold",
    "medicine_content": "Flavonoids, carotenoids, essential oils",
    "diseases_cured": ["Anti-inflammatory", "Skin conditions", "Eye health"],
    "age_restrictions": "Generally safe for all ages",
    "gender_restrictions": "No gender restrictions",
    "pregnant_use_restriction": "Consult a healthcare professional during pregnancy",
    "mode_of_use": "Flowers used in teas, infusions, or applied topically for skin care",
    "doses_for_day": "Dosage varies based on application; commonly used in moderate amounts for teas or skin treatments"
},
{
    "plant_name": "Mint",
    "medicine_content": "Menthol, flavonoids, vitamins (A, C)",
    "diseases_cured": ["Digestive health", "Respiratory conditions", "Anti-inflammatory"],
    "age_restrictions": "Generally safe for all ages",
    "gender_restrictions": "No gender restrictions",
    "pregnant_use_restriction": "Generally safe during pregnancy",
    "mode_of_use": "Leaves used in teas, infusions, culinary dishes, or as a flavoring agent",
    "doses_for_day": "Dosage varies based on culinary preferences; commonly consumed in moderate amounts"
},
{
    "plant_name": "Neem",
    "medicine_content": "Azadirachtin, nimbin, nimbidin",
    "diseases_cured": ["Anti-inflammatory", "Antibacterial", "Skin conditions"],
    "age_restrictions": "Consult a healthcare professional for appropriate use in children",
    "gender_restrictions": "No gender restrictions",
    "pregnant_use_restriction": "Avoid during pregnancy; it is harmful when taken by mouth during pregnancy",
    "mode_of_use": "Leaves, oil, and bark used in various forms, such as teas, extracts, or topical applications",
    "doses_for_day": "Dosage may vary based on application; follow the advice of a healthcare professional or herbalist"
},
{
    "plant_name": "Nelavembu",
    "medicine_content": "Alkaloids, flavonoids, terpenoids",
    "diseases_cured": ["Anti-inflammatory", "Fever", "Respiratory conditions"],
    "age_restrictions": "Generally safe for adults; consult a healthcare professional for children",
    "gender_restrictions": "No gender restrictions",
    "pregnant_use_restriction": "Avoid use during pregnancy ",
    "mode_of_use": "Leaves and stems used in decoctions or as directed by a traditional healer",
    "doses_for_day": "Dosage may vary; follow the guidance of a traditional healer or healthcare professional"
},
{
    "plant_name": "Nerale",
    "medicine_content": "Anthocyanins, ellagic acid, vitamins (C, A)",
    "diseases_cured": ["Blood sugar regulation", "Anti-inflammatory", "Digestive health"],
    "age_restrictions": "Generally safe for all ages",
    "gender_restrictions": "No gender restrictions",
    "pregnant_use_restriction": "Generally safe during pregnancy as it contains antioxidants and nutrients which is beneficial for foetal developments ",
    "mode_of_use": "Fruit consumed fresh, in juices, or used in culinary dishes",
    "doses_for_day": "Dosage varies based on culinary preferences; commonly consumed in moderate amounts"
},
{
    "plant_name": "Nooni",
    "medicine_content": "Vitamins (A, C), dietary fiber, minerals",
    "diseases_cured": ["Digestive health", "Immune system support", "Skin conditions"],
    "age_restrictions": "Generally safe for all ages",
    "gender_restrictions": "No gender restrictions",
    "pregnant_use_restriction": "Consult a healthcare professional during pregnancy",
    "mode_of_use": "Fruit consumed fresh or dried, often used in culinary dishes",
    "doses_for_day": "Dosage varies based on culinary preferences; commonly consumed in moderate amounts"
},
{
    "plant_name": "Onion",
    "medicine_content": "Quercetin, sulfur compounds, vitamins (C, B6)",
    "diseases_cured": ["Anti-inflammatory", "Cardiovascular health", "Immune system support"],
    "age_restrictions": "Safe for all ages",
    "gender_restrictions": "No gender restrictions",
    "pregnant_use_restriction": "Generally safe during pregnancy as it contains antioxidants, vitamin C, vitamin B6 and trace minerals",
    "mode_of_use": "Bulbs used in culinary dishes, raw or cooked",
    "doses_for_day": "Dosage varies based on culinary preferences; commonly consumed in moderate amounts"
},
{
    "plant_name": "Padri",
    "medicine_content": "Phytochemicals, flavonoids, tannins",
    "diseases_cured": ["Anti-inflammatory", "Fever", "Respiratory conditions"],
    "age_restrictions": "Consult a healthcare professional for appropriate use in children",
    "gender_restrictions": "No gender restrictions",
    "pregnant_use_restriction": "Consult a healthcare professional during pregnancy",
    "mode_of_use": "Bark and leaves used in decoctions or as directed by a traditional healer",
    "doses_for_day": "Dosage may vary; follow the guidance of a traditional healer or healthcare professional"
},
{
    "plant_name": "Palak (Spinach)",
    "medicine_content": "Vitamins (A, C, K), iron, antioxidants",
    "diseases_cured": ["Improves hemoglobin levels", "Bone health", "Anti-inflammatory"],
    "age_restrictions": "Safe for all ages",
    "gender_restrictions": "No gender restrictions",
    "pregnant_use_restriction": "Generally safe during pregnancy",
    "mode_of_use": "Leaves used in cooking, salads, or as part of various dishes",
    "doses_for_day": "Dosage varies based on culinary preferences; commonly consumed in moderate amounts"
},
{
    "plant_name": "Papaya",
    "medicine_content": "Papain, vitamins (C, A), dietary fiber",
    "diseases_cured": ["Digestive health", "Immune system support", "Skin conditions"],
    "age_restrictions": "Safe for all ages",
    "gender_restrictions": "No gender restrictions",
    "pregnant_use_restriction": "Riped papaya is safe during pregnancy, if the papaya is green or semi-riped it should not be used ",
    "mode_of_use": "Fruit consumed fresh, in smoothies, or used in culinary dishes",
    "doses_for_day": "Dosage varies based on culinary preferences; commonly consumed in moderate amounts"
},
{
    "plant_name": "Parijatha",
    "medicine_content": "Flavonoids, essential oils",
    "diseases_cured": ["Anti-inflammatory", "Stress relief", "Skin conditions"],
    "age_restrictions": "Generally safe for all ages",
    "gender_restrictions": "No gender restrictions",
    "pregnant_use_restriction": "Consult a healthcare professional during pregnancy",
    "mode_of_use": "Flowers and leaves used in teas, infusions, or as decorative elements",
    "doses_for_day": "Dosage varies based on application; commonly used in moderate amounts for teas or decorative purposes"
},
{
    "plant_name": "Pea",
    "medicine_content": "Protein, fiber, vitamins (C, K), antioxidants",
    "diseases_cured": ["Digestive health", "Heart health", "Immune system support"],
    "age_restrictions": "Safe for all ages",
    "gender_restrictions": "No gender restrictions",
    "pregnant_use_restriction": "Generally safe during pregnancy, but consult a nutritionist",
    "mode_of_use": "Pods and peas used in cooking, salads, or as part of various dishes",
    "doses_for_day": "Dosage varies based on culinary preferences; commonly consumed in moderate amounts"
},
{
    "plant_name": "Pepper",
    "medicine_content": "Piperine, vitamins (C, K), antioxidants",
    "diseases_cured": ["Digestive health", "Anti-inflammatory", "Antioxidant properties"],
    "age_restrictions": "Safe for all ages",
    "gender_restrictions": "No gender restrictions",
    "pregnant_use_restriction": "Generally safe during pregnancy, but consult a healthcare professional",
    "mode_of_use": "Peppercorns used as a spice in cooking or as a seasoning",
    "doses_for_day": "Dosage varies based on culinary preferences; commonly used in moderate amounts"
},
{
    "plant_name": "Pomegranate",
    "medicine_content": "Punicalagins, anthocyanins, vitamins (C, K)",
    "diseases_cured": ["Heart health", "Antioxidant properties", "Anti-inflammatory"],
    "age_restrictions": "Safe for all ages",
    "gender_restrictions": "No gender restrictions",
    "pregnant_use_restriction": "Generally safe during pregnancy,",
    "mode_of_use": "Seeds and juice consumed fresh, in smoothies, or used in culinary dishes",
    "doses_for_day": "Dosage varies based on culinary preferences; commonly consumed in moderate amounts"
},
{
    "plant_name": "Pumpkin",
    "medicine_content": "Beta-carotene, vitamins (A, C), fiber",
    "diseases_cured": ["Eye health", "Immune system support", "Heart health"],
    "age_restrictions": "Safe for all ages",
    "gender_restrictions": "No gender restrictions",
    "pregnant_use_restriction": "Generally safe during pregnancy, but consult a nutritionist",
    "mode_of_use": "Flesh and seeds used in cooking, soups, or as part of various dishes",
    "doses_for_day": "Dosage varies based on culinary preferences; commonly consumed in moderate amounts"
},
{
    "plant_name": "Radish",
    "medicine_content": "Vitamins (C, K), fiber, antioxidants",
    "diseases_cured": ["Digestive health", "Immune system support", "Detoxification"],
    "age_restrictions": "Safe for all ages",
    "gender_restrictions": "No gender restrictions",
    "pregnant_use_restriction": "Generally safe during pregnancy, but consult a nutritionist",
    "mode_of_use": "Roots and leaves used in salads, cooking, or as a crunchy snack",
    "doses_for_day": "Dosage varies based on culinary preferences; commonly consumed in moderate amounts"
},
{
    "plant_name": "Rose",
    "medicine_content": "Essential oils, flavonoids, vitamins (C)",
    "diseases_cured": ["Stress relief", "Skin conditions", "Anti-inflammatory"],
    "age_restrictions": "Generally safe for all ages",
    "gender_restrictions": "No gender restrictions",
    "pregnant_use_restriction": "Consult a healthcare professional during pregnancy",
    "mode_of_use": "Flowers and petals used in teas, infusions, or as decorative elements",
    "doses_for_day": "Dosage varies based on application; commonly used in moderate amounts for teas or decorative purposes"
},
{
    "plant_name": "Sampige",
    "medicine_content": "Essential oils, flavonoids",
    "diseases_cured": ["Stress relief", "Anti-inflammatory", "Skin conditions"],
    "age_restrictions": "Generally safe for all ages",
    "gender_restrictions": "No gender restrictions",
    "pregnant_use_restriction": "Consult a healthcare professional during pregnancy",
    "mode_of_use": "Flowers and leaves used in teas, infusions, or as decorative elements",
    "doses_for_day": "Dosage varies based on application; commonly used in moderate amounts for teas or decorative purposes"
},
{
    "plant_name": "Sapota",
    "medicine_content": "Vitamins (A, C), dietary fiber, antioxidants",
    "diseases_cured": ["Digestive health", "Boosts immunity", "Heart health"],
    "age_restrictions": "Safe for all ages",
    "gender_restrictions": "No gender restrictions",
    "pregnant_use_restriction": "Generally safe during pregnancy, but consult a nutritionist",
    "mode_of_use": "Fruit consumed fresh, in smoothies, or used in culinary dishes",
    "doses_for_day": "Dosage varies based on culinary preferences; commonly consumed in moderate amounts"
},
{
    "plant_name": "Seethaashoka",
    "medicine_content": "Flavonoids, alkaloids",
    "diseases_cured": ["Anti-inflammatory", "Menstrual disorders", "Stress relief"],
    "age_restrictions": "Consult a healthcare professional for appropriate use in children",
    "gender_restrictions": "Commonly used for women's health",
    "pregnant_use_restriction": "Consult a healthcare professional during pregnancy",
    "mode_of_use": "Bark and flowers used in decoctions or as directed by a traditional healer",
    "doses_for_day": "Dosage may vary; follow the guidance of a traditional healer or healthcare professional"
},
{
    "plant_name": "Seethapala",
    "medicine_content": "Vitamins (C, B6), dietary fiber, minerals",
    "diseases_cured": ["Digestive health", "Immune system support", "Heart health"],
    "age_restrictions": "Generally safe for all ages",
    "gender_restrictions": "No gender restrictions",
    "pregnant_use_restriction": "Generally safe during pregnancy, but consult a nutritionist",
    "mode_of_use": "Fruit consumed fresh or in desserts",
    "doses_for_day": "Dosage varies based on culinary preferences; commonly consumed in moderate amounts"
},
{
    "plant_name": "Spinach1",
    "medicine_content": "Vitamins (A, C, K), iron, antioxidants",
    "diseases_cured": ["Improves hemoglobin levels", "Bone health", "Anti-inflammatory"],
    "age_restrictions": "Safe for all ages",
    "gender_restrictions": "No gender restrictions",
    "pregnant_use_restriction": "Generally safe during pregnancy, but consult a nutritionist",
    "mode_of_use": "Leaves used in cooking, salads, or as part of various dishes",
    "doses_for_day": "Dosage varies based on culinary preferences; commonly consumed in moderate amounts"
},
{
    "plant_name": "Tamarind",
    "medicine_content": "Tartaric acid, vitamins (C, B), antioxidants",
    "diseases_cured": ["Digestive health", "Anti-inflammatory", "Antioxidant properties"],
    "age_restrictions": "Safe for all ages",
    "gender_restrictions": "No gender restrictions",
    "pregnant_use_restriction": "Generally safe during pregnancy, but consult a nutritionist",
    "mode_of_use": "Pulp and seeds used in culinary dishes, sauces, or as a flavoring agent",
    "doses_for_day": "Dosage varies based on culinary preferences; commonly consumed in moderate amounts"
},
{
    "plant_name": "Taro",
    "medicine_content": "Dietary fiber, vitamins (E, B6), minerals",
    "diseases_cured": ["Digestive health", "Heart health", "Anti-inflammatory"],
    "age_restrictions": "Safe for all ages",
    "gender_restrictions": "No gender restrictions",
    "pregnant_use_restriction": "Generally safe during pregnancy, but consult a nutritionist",
    "mode_of_use": "Roots and leaves used in cooking, often boiled, fried, or used in stews",
    "doses_for_day": "Dosage varies based on culinary preferences; commonly consumed in moderate amounts"
},
{
    "plant_name": "Tecoma",
    "medicine_content": "Flavonoids, alkaloids",
    "diseases_cured": ["Anti-inflammatory", "Digestive disorders", "Immune system support"],
    "age_restrictions": "Consult a healthcare professional for appropriate use in children",
    "gender_restrictions": "No gender restrictions",
    "pregnant_use_restriction": "Consult a healthcare professional during pregnancy",
    "mode_of_use": "Flowers and leaves used in teas, infusions, or as decorative elements",
    "doses_for_day": "Dosage varies based on application; commonly used in moderate amounts for teas or decorative purposes"
},
{
    "plant_name": "Thumbe",
    "medicine_content": "Flavonoids, alkaloids",
    "diseases_cured": ["Anti-inflammatory", "Digestive disorders", "Immune system support"],
    "age_restrictions": "Generally safe for all ages",
    "gender_restrictions": "No gender restrictions",
    "pregnant_use_restriction": "Consult a healthcare professional during pregnancy",
    "mode_of_use": "Flowers and leaves used in teas, infusions, or as decorative elements",
    "doses_for_day": "Dosage varies based on application; commonly used in moderate amounts for teas or decorative purposes"
},
{
    "plant_name": "Tomato",
    "medicine_content": "Lycopene, vitamins (C, K), antioxidants",
    "diseases_cured": ["Heart health", "Anti-inflammatory", "Immune system support"],
    "age_restrictions": "Safe for all ages",
    "gender_restrictions": "No gender restrictions",
    "pregnant_use_restriction": "Generally safe during pregnancy, but consult a nutritionist",
    "mode_of_use": "Fruit used in salads, sauces, or consumed raw",
    "doses_for_day": "Dosage varies based on culinary preferences; commonly consumed in moderate amounts"
},
{
    "plant_name": "Tulsi",
    "medicine_content": "Ocimumosides, eugenol, essential oils",
    "diseases_cured": ["Anti-inflammatory", "Respiratory conditions", "Stress relief"],
    "age_restrictions": "Generally safe for all ages",
    "gender_restrictions": "No gender restrictions",
    "pregnant_use_restriction": "Not safe during pregnancy as it stimulates the flow of blood to uterus and pelvis area which can cause contractions",
    "mode_of_use": "Leaves used in teas, infusions, or consumed raw",
    "doses_for_day": "Dosage varies based on application; commonly consumed in moderate amounts as tea or in culinary dishes"
},
{
    "plant_name": "Turmeric",
    "medicine_content": "Curcumin, vitamins (C, B6), antioxidants",
    "diseases_cured": ["Anti-inflammatory", "Digestive health", "Antioxidant properties"],
    "age_restrictions": "Safe for all ages",
    "gender_restrictions": "No gender restrictions",
    "pregnant_use_restriction": "Generally safe during pregnancy, but consult a healthcare professional",
    "mode_of_use": "Rhizomes used in cooking, teas, or as a spice",
    "doses_for_day": "Dosage varies based on culinary preferences; commonly consumed in moderate amounts"
},
{
    "plant_name": "Ashoka",
    "medicine_content": "Flavonoids, alkaloids",
    "diseases_cured": ["Anti-inflammatory", "Menstrual disorders", "Stress relief"],
    "age_restrictions": "Consult a healthcare professional for appropriate use in children",
    "gender_restrictions": "Commonly used for women's health",
    "pregnant_use_restriction": "Consult a healthcare professional during pregnancy",
    "mode_of_use": "Bark and flowers used in decoctions or as directed by a traditional healer",
    "doses_for_day": "Dosage may vary; follow the guidance of a traditional healer or healthcare professional"
},
{
    "plant_name": "Camphor",
    "medicine_content": "Camphor",
    "diseases_cured": ["Respiratory conditions", "Topical analgesic", "Insect repellent"],
    "age_restrictions": "Consult a healthcare professional for appropriate use in children",
    "gender_restrictions": "No gender restrictions",
    "pregnant_use_restriction": "Avoid during pregnancy; consult a healthcare professional",
    "mode_of_use": "Used in small quantities in inhalation, topical applications, or as an insect repellent",
    "doses_for_day": "Dosage varies based on application; use in moderation"
},
{
    "plant_name": "Kamakasturi",
    "medicine_content": "Essential oils, flavonoids",
    "diseases_cured": ["Anti-inflammatory", "Digestive disorders", "Stress relief"],
    "age_restrictions": "Generally safe for all ages",
    "gender_restrictions": "No gender restrictions",
    "pregnant_use_restriction": "Not safe during pregnancy; consult healthcare specialist",
    "mode_of_use": "Leaves used in culinary dishes, teas, or as a garnish",
    "doses_for_day": "Dosage varies based on culinary preferences; commonly consumed in moderate amounts"
},
{
    "plant_name": "Kepala",
    "medicine_content": "Flavonoids, tannins",
    "diseases_cured": ["Anti-inflammatory", "Digestive disorders", "Skin conditions"],
    "age_restrictions": "Generally safe for all ages",
    "gender_restrictions": "No gender restrictions",
    "pregnant_use_restriction": "Consult a healthcare professional during pregnancy",
    "mode_of_use": "Flowers and leaves used in teas, infusions, or as decorative elements",
    "doses_for_day": "Dosage varies based on application; commonly used in moderate amounts for teas or decorative purposes"
}
]





disease_dic= [
    "Aloevera",
    "Amla",
    "Amruthaballi",
    "Arali",
    "Astma_weed",
    "Badipala",
    "Balloon_Vine",
    "Bamboo",
    "Beans",
    "Betel",
    "Bhrami",
    "Bringaraja",
    "Caricature",
    "Castor",
    "Catharanthus",
    "Chakte",
    "Chilly",
    "Citron lime (herelikai)",
    "Coffee",
    "Common rue(naagdalli)",
    "Coriander",
    "Curry",
    "Doddpathre",
    "Drumstick",
    "Ekka",
    "Eucalyptus",
    "Ganigale",
    "Ganike",
    "Gasagase",
    "Ginger",
    "Globe Amaranth",
    "Guava",
    "Henna",
    "Hibiscus",
    "Honge",
    "Insulin",
    "Jackfruit",
    "Jasmine",
    "Kambajala",
    "Kasambruga",
    "Kohlrabi",
    "Lantana",
    "Lemon",
    "Lemongrass",
    "Malabar Nut",
    "Malabar Spinach",
    "Mango",
    "Marigold",
    "Mint",
    "Neem",
    "Nelavembu",
    "Nerale",
    "Nooni",
    "Onion",
    "Padri",
    "Palak (Spinach)",
    "Papaya",
    "Parijatha",
    "Pea",
    "Pepper",
    "Pomegranate",
    "Pumpkin",
    "Radish",
    "Rose",
    "Sampige",
    "Sapota",
    "Seethaashoka",
    "Seethapala",
    "Spinach1",
    "Tamarind",
    "Taro",
    "Tecoma",
    "Thumbe",
    "Tomato",
    "Tulsi",
    "Turmeric",
    "Ashoka",
    "Camphor",
    "Kamakasturi",
    "Kepala"
]

details = None

from model_predict  import pred_leaf_disease

# ===============================================================================================
# ------------------------------------ FLASK APP -------------------------------------------------


app = Flask(__name__)
# render home page


@ app.route('/')
def home():
    #title = 'Medicinal plant Detection'
    return render_template('index3.html')



@ app.route('/index2')
def index2():
    #title = 'Medicinal plant Detection'
    return render_template('main_page.html')


@ app.route('/index3')
def index3():
    #title = 'Medicinal plant Detection'
    return render_template('index3.html')


@ app.route('/contact', methods=['GET','POST'])
def contact():
    
    if request.method == 'POST':
    #title = 'Medicinal plant Detection'
    
        name = request.form['name']
        email = request.form['email']
        message = request.form['message']

        # Save data to a file
        with open('form_data.txt', 'a') as f:
            f.write(f'Name: {name}\nEmail: {email}\nMessage: {message}\n\n')
        
        return redirect('index3')

    return render_template('contact.html')







@app.route('/disease-predict', methods=['GET', 'POST'])
def disease_prediction():
    title = 'Diabetic Retinopathy Detection'

    if request.method == 'POST':
        #if 'file' not in request.files:
         #   return redirect(request.url)

            file = request.files.get('file')

           # if not file:
            #    return render_template('disease.html', title=title)

            img = Image.open(file)
            img.save('output.png')

            # Define the path to save the image
            output_path1 = "static/images/output.png"

            # Save the image to the specified path
            img.save(output_path1)
            prediction,prob =pred_leaf_disease("output.png")

            prediction = (str(disease_dic[prediction]))

            print(prediction)

            
            for plant in plant_data:
                if plant['plant_name'] == prediction:
                    details = plant
                    break


            return render_template('result_page.html', details=details,prob=prob)



           
         #   pass
    return render_template('disease.html', title=title)





# ===============================================================================================
if __name__ == '__main__':
    app.run(debug=True)
