
#get MedNLI
mkdir -p /MedNLI

#getRQE
mkdir -p ./RQE
cd ./RQE
wget https://raw.githubusercontent.com/abachaa/RQE_Data_AMIA2016/master/RQE_Train_8588_AMIA2016.xml
wget https://raw.githubusercontent.com/abachaa/RQE_Data_AMIA2016/master/RQE_Test_302_pairs_AMIA2016.xml
  

#get PICO
mkdir -p ../PICO
cd ../PICO
wget -O train.txt https://github.com/jind11/PubMed-PICO-Detection/blob/master/splitted/PICO_train.txt?raw=true 
wget -O dev.txt https://github.com/jind11/PubMed-PICO-Detection/blob/master/splitted/PICO_dev.txt?raw=true 
wget -O test.txt https://github.com/jind11/PubMed-PICO-Detection/blob/master/splitted/PICO_test.txt?raw=true
  
#get Pubmed20K
mkdir -p ../PubMed20K
cd ../PubMed20K
wget -O train.txt https://github.com/Franck-Dernoncourt/pubmed-rct/blob/master/PubMed_20k_RCT/train.txt?raw=true
wget -O dev.txt https://github.com/Franck-Dernoncourt/pubmed-rct/blob/master/PubMed_20k_RCT/dev.txt?raw=true
wget -O test.txt https://github.com/Franck-Dernoncourt/pubmed-rct/blob/master/PubMed_20k_RCT/test.txt?raw=true
  
#get CitaionSA
mkdir -p ../CitaionSA

#get VaccineSA
mkdir -p ../ClinicalSA
cd ../ClinicalSA
wget https://sbmi.uth.edu/ontology/files/TweetsAnnotationResults.zip
unzip TweetsAnnotationResults.zip
#java  -jar AnnotationResults/DownloadTweets.jar AnnotationResults/TweetsAnnotation.txt ./train_temp.txt >/dev/null

  
#get BioASQ
mkdir -p ../BioASQ
 
#get BIOC
mkdir -p ../BIOC
cd ../BIOC
wget http://staffwww.dcs.shef.ac.uk/people/M.Stevenson/resources/bio_contradictions/corpus.xml

#get ClinicalSTS
mkdir -p ../ClinicalSTS

#get BIOSSES
mkdir -p ../BIOSSES
cd ../BIOSSES
wget https://bitbucket.org/gizemsogancioglu/biosses-resources/raw/52a77008d6c80ea570fa717136421b8c81683aa2/resources.zip
unzip -p resources.zip correlationResult/groundTruth/test.txt > STS.gs.BIOSSES.txt
unzip -p resources.zip sentencePairsData/pairs.txt > temp.txt
awk 'BEGIN{FS="\t"}{printf ("%s\t%s\n", $2, $3)}' temp.txt > STS.input.BIOSSES.txt

#prepare data files
python ../prepare_data.py
