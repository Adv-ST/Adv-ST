mkdir voxpopuli_es
mkdir voxpopuli_es/2009
cp ~/data/voxpopuli/download/raw_audios/es/2009/20091215-0900-PLENARY-19_es.ogg voxpopuli_es/2009/

mkdir voxpopuli_es/2011
cp ~/data/voxpopuli/download/raw_audios/es/2011/20110707-0900-PLENARY-15_es.ogg voxpopuli_es/2011/


mkdir -p voxpopuli_fr/2009 voxpopuli_fr/2011
cp ~/data/voxpopuli/download/raw_audios/fr/2009/20091215-0900-PLENARY-19_fr.ogg voxpopuli_fr/2009/
cp ~/data/voxpopuli/download/raw_audios/fr/2011/20110216-0900-PLENARY-18_fr.ogg voxpopuli_fr/2011/

mkdir -p voxpopuli_de/2009 voxpopuli_de/2011
cp ~/data/voxpopuli/download/raw_audios/de/2009/20091215-0900-PLENARY-19_de.ogg voxpopuli_de/2009/
cp ~/data/voxpopuli/download/raw_audios/de/2011/20110216-0900-PLENARY-18_de.ogg voxpopuli_de/2011/

mkdir -p voxpopuli_it/2009 voxpopuli_it/2011
cp ~/data/voxpopuli/download/raw_audios/it/2009/20091215-0900-PLENARY-19_it.ogg voxpopuli_it/2009/
cp ~/data/voxpopuli/download/raw_audios/it/2011/20110707-0900-PLENARY-15_it.ogg voxpopuli_it/2011/

python ~/data/voxpopuli/raw_audios/to_wav.py