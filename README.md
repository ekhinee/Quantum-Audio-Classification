# Phase-Sensitive-Quantum-Time-Series-Encoding-For-Audio-Classification


Proiektua osatzen duten fitxategiak:

## datuak_kargatu.py

Dituen funtzioetatik **save_audios_gtzan** eta **prep_folds** datuak preparatzeko eta gordetzeko erabili ditut

Pipeline-an load_audios baino ez da erabiltzen audioak kargatzeko beharrezkoa den Path-etik

## preprozesaketa.py

Funtzio nagusiak **preprocess** eta **normalize** dira (hauek deitzen dira pipeline-tik)

### preprocess

Datu gordinak hartu eta aurreprozesaketa aplikatzen zaio, hiru motatakoa izan daiteke (*type*): "mfcc", "phase", eta "gd". Gainera pipeline klasiko edo kuantikoa izango den ala ez zehaztu daiteke (*classic*: Boolean), kuantikoa bada itzulitako balioak 16-ko bektoretan egongo dira seinaleko.

- "mfcc" edo "phase" motakoa bada **preprocess_mfcc** funtzioari deitzen dio. Bertan mfcc klasikoak edo faseari dagozkionak kalkulatzen dira, azken honetan **mfcc_phase** funtzioaz baliatuz.

- "gd" motakoa bada **preproc_gropup_delay** funtzioari deitzen dio, zeinak seinaleko dagozkion group delay balioak kalkulatzen dituen. Horretarako seinale bakoitzeko **MODGDF_signal** funtzioari deitzen dio. Bertan subframe bakoitzeko **MODGDF** funtzioaren bitartez group delay-a kalkulatzen da.

## kodeketak.py

Fitxategi honetan **build_feature_map** funtzioa da printzipala, zeinak kodeketa mota (*feature_map*) eta entrenatu behar den edo ez (*train*) jasota dagokion kodeketa funtzioari deitzen dion. Kodeketa funtzioak honakoak dira:
- qtse
- qtse_trainable
- qtse_p1
- qtse_p2
- qtse_p2_trainable
- ryrz_1: 5 qubit (4 denbora, 1 datuak)
- ryrz_1_trainable:
- qtse_p3: ryrz-ren idea berdina baina qtse moduan kodetzen du denbora (8 kubit: 4 denbora, 4 datuak)
- qtse_p3_trainable

## kernel.py

**get_kernel_matrix** da funtzio printzipala, eta bi funtzio laguntzaile erabiltzen ditu: **get_fidelity** eta **fidelity_circuit**.

### get_kernel_matrix

Honen bidez kalkulatzen da train edo train-test arteko Kernel matrizea. Pare bakoitzeko fidelity-a lortzen da **get_fidelity** funtzioaren bidez, zeinak behar den kernel zirkuitua sortzen duen **fidelity_circuit** erabiliz eta statevecto bidez kalkulatzen duen fidelity-a.

## svm.py

Hemen svm erabiltzeko beharrezko funtzioak daude; zehazki klasikorako **svm_classic_kernel**, eta kuantikorako **svm_classification**.

## exp.py

Hemen pipeline osoa eraikitzeko funtzioa dago: **pipeline_cluster**. Parametroen arabera dagokion esperimentazio konfigurazioa martxan jartzen du. Parametroei dagokienez:

- dataset_path: string.
- classic: Boolean. Pipeline klasikoa jarraituko badu True izango da, bestela False
- preproc_mfcc: Boolean. Aurreprozesaketa mfcc-ak ateratzea behar baditu True, bestela False.
- train: Boolean.  Encoding-a entrenatu egingo bada True, bestela False.
- preproc_phase: Boolean. Aurreprozesaketan fase informazioa ateratzea behar badu treu, bestela false.
- phse_type: string. preproc_phase True izanez gero, hemen espezifikatu behar da zein motakoa: "phase" (fase gordinak hartu)  edo "gd" (group delay) balioak har ditzake.
- encoding: string. Erabiliko den kodeketa zehazteko, aukerak: "qtse", "qtse_p1", "qtse_p2", "ryrz_1", eta "qtse_p3" ("qtse_p1" ez dago entrenatzerik)
- init_w: float. Train True izanez gero, w parametroen hasieraketa
- init_b: float. Train True izanez gero, b parametroen hasieraketa

## main.py

Hemen parametroak zehaztu eta entrenamendua gauzatzen da. 

