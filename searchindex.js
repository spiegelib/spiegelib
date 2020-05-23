Search.setIndex({docnames:["contributing","examples/examples_index","examples/fm_sound_match","examples/fm_sound_match_pages/fm_sound_match_dataset_generation","examples/fm_sound_match_pages/fm_sound_match_deep_learning","examples/fm_sound_match_pages/fm_sound_match_evaluation","examples/fm_sound_match_pages/fm_sound_match_genetic","examples/fm_sound_match_pages/fm_sound_match_listen","examples/fm_sound_match_pages/fm_sound_match_synth_config","examples/fm_sound_match_pages/fm_sound_match_train_models","getting_started/getting_started","getting_started/installation","index","reference/core","reference/core/audio_buffer","reference/core/dataset_generator","reference/core/sound_match","reference/core/utils","reference/estimators","reference/estimators/basic_ga","reference/estimators/conv6","reference/estimators/estimator_base","reference/estimators/highway_layer","reference/estimators/hwy_blstm","reference/estimators/lstm","reference/estimators/mlp","reference/estimators/nsga3","reference/estimators/tf_epoch_logger","reference/estimators/tf_estimator_base","reference/evaluation","reference/evaluation/evaluation_base","reference/evaluation/mfcc_eval","reference/features","reference/features/data_scaler_base","reference/features/features_base","reference/features/fft","reference/features/mfcc","reference/features/spectral_summarized","reference/features/standard_scaler","reference/features/stft","reference/synth","reference/synth/synth_base","reference/synth/synth_vst"],envversion:{"sphinx.domains.c":1,"sphinx.domains.changeset":1,"sphinx.domains.citation":1,"sphinx.domains.cpp":1,"sphinx.domains.index":1,"sphinx.domains.javascript":1,"sphinx.domains.math":2,"sphinx.domains.python":1,"sphinx.domains.rst":1,"sphinx.domains.std":1,sphinx:56},filenames:["contributing.rst","examples/examples_index.rst","examples/fm_sound_match.rst","examples/fm_sound_match_pages/fm_sound_match_dataset_generation.rst","examples/fm_sound_match_pages/fm_sound_match_deep_learning.rst","examples/fm_sound_match_pages/fm_sound_match_evaluation.rst","examples/fm_sound_match_pages/fm_sound_match_genetic.rst","examples/fm_sound_match_pages/fm_sound_match_listen.rst","examples/fm_sound_match_pages/fm_sound_match_synth_config.rst","examples/fm_sound_match_pages/fm_sound_match_train_models.rst","getting_started/getting_started.rst","getting_started/installation.rst","index.rst","reference/core.rst","reference/core/audio_buffer.rst","reference/core/dataset_generator.rst","reference/core/sound_match.rst","reference/core/utils.rst","reference/estimators.rst","reference/estimators/basic_ga.rst","reference/estimators/conv6.rst","reference/estimators/estimator_base.rst","reference/estimators/highway_layer.rst","reference/estimators/hwy_blstm.rst","reference/estimators/lstm.rst","reference/estimators/mlp.rst","reference/estimators/nsga3.rst","reference/estimators/tf_epoch_logger.rst","reference/estimators/tf_estimator_base.rst","reference/evaluation.rst","reference/evaluation/evaluation_base.rst","reference/evaluation/mfcc_eval.rst","reference/features.rst","reference/features/data_scaler_base.rst","reference/features/features_base.rst","reference/features/fft.rst","reference/features/mfcc.rst","reference/features/spectral_summarized.rst","reference/features/standard_scaler.rst","reference/features/stft.rst","reference/synth.rst","reference/synth/synth_base.rst","reference/synth/synth_vst.rst"],objects:{"spiegelib.AudioBuffer":{get_audio:[14,1,1,""],get_sample_rate:[14,1,1,""],load:[14,1,1,""],load_folder:[14,1,1,""],peak_normalize:[14,1,1,""],plot_spectrogram:[14,1,1,""],replace_audio_data:[14,1,1,""],resize:[14,1,1,""],save:[14,1,1,""]},"spiegelib.DatasetGenerator":{audio_folder_name:[15,2,1,""],features_filename:[15,2,1,""],generate:[15,1,1,""],patches_filename:[15,2,1,""],save_scaler:[15,1,1,""]},"spiegelib.SoundMatch":{get_patch:[16,1,1,""],match:[16,1,1,""],match_from_file:[16,1,1,""]},"spiegelib.core":{audio_buffer:[14,3,0,"-"],dataset_generator:[15,3,0,"-"],sound_match:[16,3,0,"-"],utils:[17,3,0,"-"]},"spiegelib.core.utils":{NumpyNumberEncoder:[17,0,1,""],atoi:[17,4,1,""],convert_spectrum:[17,4,1,""],natural_keys:[17,4,1,""],spectrum_types:[17,5,1,""]},"spiegelib.estimator":{BasicGA:[19,0,1,""],Conv6:[20,0,1,""],HighwayLayer:[22,0,1,""],HwyBLSTM:[23,0,1,""],LSTM:[24,0,1,""],MLP:[25,0,1,""],NSGA3:[26,0,1,""],TFEpochLogger:[27,0,1,""],TFEstimatorBase:[28,0,1,""],basic_ga:[19,3,0,"-"],conv6:[20,3,0,"-"],estimator_base:[21,3,0,"-"],highway_layer:[22,3,0,"-"],hwy_blstm:[23,3,0,"-"],lstm:[24,3,0,"-"],mlp:[25,3,0,"-"],nsga3:[26,3,0,"-"],tf_epoch_logger:[27,3,0,"-"],tf_estimator_base:[28,3,0,"-"]},"spiegelib.estimator.BasicGA":{fitness:[19,1,1,""],predict:[19,1,1,""]},"spiegelib.estimator.Conv6":{build_model:[20,1,1,""]},"spiegelib.estimator.HighwayLayer":{build:[22,1,1,""],call:[22,1,1,""],compute_output_shape:[22,1,1,""],get_config:[22,1,1,""]},"spiegelib.estimator.HwyBLSTM":{build_model:[23,1,1,""]},"spiegelib.estimator.LSTM":{build_model:[24,1,1,""]},"spiegelib.estimator.MLP":{build_model:[25,1,1,""]},"spiegelib.estimator.NSGA3":{fitness:[26,1,1,""],predict:[26,1,1,""]},"spiegelib.estimator.TFEpochLogger":{get_plotting_data:[27,1,1,""],log_data:[27,2,1,""],on_epoch_end:[27,1,1,""],plot:[27,1,1,""]},"spiegelib.estimator.TFEstimatorBase":{add_testing_data:[28,1,1,""],add_training_data:[28,1,1,""],build_model:[28,1,1,""],fit:[28,1,1,""],load:[28,1,1,""],load_weights:[28,1,1,""],model:[28,2,1,""],predict:[28,1,1,""],rms_error:[28,1,1,""],save_model:[28,1,1,""],save_weights:[28,1,1,""]},"spiegelib.estimator.estimator_base":{EstimatorBase:[21,0,1,""]},"spiegelib.estimator.estimator_base.EstimatorBase":{predict:[21,1,1,""]},"spiegelib.evaluation":{EvaluationBase:[30,0,1,""],MFCCEval:[31,0,1,""],evaluation_base:[30,3,0,"-"],mfcc_eval:[31,3,0,"-"]},"spiegelib.evaluation.EvaluationBase":{euclidian_distance:[30,1,1,""],evaluate:[30,1,1,""],evaluate_target:[30,1,1,""],get_scores:[30,1,1,""],get_stats:[30,1,1,""],manhattan_distance:[30,1,1,""],mean_abs_error:[30,1,1,""],mean_squared_error:[30,1,1,""],plot_hist:[30,1,1,""],save_scores_json:[30,1,1,""],save_stats_json:[30,1,1,""],verify_audio_input_list:[30,1,1,""],verify_input_list:[30,1,1,""]},"spiegelib.evaluation.MFCCEval":{evaluate_target:[31,1,1,""],verify_input_list:[31,1,1,""]},"spiegelib.features":{DataScalerBase:[33,0,1,""],FFT:[35,0,1,""],FeaturesBase:[34,0,1,""],MFCC:[36,0,1,""],STFT:[39,0,1,""],SpectralSummarized:[37,0,1,""],StandardScaler:[38,0,1,""],data_scaler_base:[33,3,0,"-"],features_base:[34,3,0,"-"],fft:[35,3,0,"-"],mfcc:[36,3,0,"-"],spectral_summarized:[37,3,0,"-"],standard_scaler:[38,3,0,"-"],stft:[39,3,0,"-"]},"spiegelib.features.DataScalerBase":{fit:[33,1,1,""],fit_transform:[33,1,1,""],transform:[33,1,1,""]},"spiegelib.features.FFT":{get_features:[35,1,1,""]},"spiegelib.features.FeaturesBase":{__call__:[34,1,1,""],add_modifier:[34,1,1,""],fit_scaler:[34,1,1,""],get_features:[34,1,1,""],has_scaler:[34,1,1,""],load_scaler:[34,1,1,""],save_scaler:[34,1,1,""],scale:[34,1,1,""],set_scaler:[34,1,1,""]},"spiegelib.features.MFCC":{get_features:[36,1,1,""]},"spiegelib.features.STFT":{get_features:[39,1,1,""]},"spiegelib.features.SpectralSummarized":{get_features:[37,1,1,""]},"spiegelib.features.StandardScaler":{fit:[38,1,1,""],transform:[38,1,1,""]},"spiegelib.synth":{SynthBase:[41,0,1,""],SynthVST:[42,0,1,""],synth_base:[41,3,0,"-"],synth_vst:[42,3,0,"-"]},"spiegelib.synth.SynthBase":{get_audio:[41,1,1,""],get_parameters:[41,1,1,""],get_patch:[41,1,1,""],get_random_example:[41,1,1,""],load_patch:[41,1,1,""],load_state:[41,1,1,""],randomize_patch:[41,1,1,""],render_patch:[41,1,1,""],save_state:[41,1,1,""],set_overridden_parameters:[41,1,1,""],set_patch:[41,1,1,""]},"spiegelib.synth.SynthVST":{get_audio:[42,1,1,""],get_parameters:[42,1,1,""],get_patch:[42,1,1,""],get_random_example:[42,1,1,""],is_valid_parameter_setting:[42,1,1,""],load_patch:[42,1,1,""],load_plugin:[42,1,1,""],load_state:[42,1,1,""],randomize_patch:[42,1,1,""],render_patch:[42,1,1,""],save_state:[42,1,1,""],set_overridden_parameters:[42,1,1,""],set_patch:[42,1,1,""]},spiegelib:{AudioBuffer:[14,0,1,""],DatasetGenerator:[15,0,1,""],SoundMatch:[16,0,1,""]}},objnames:{"0":["py","class","Python class"],"1":["py","method","Python method"],"2":["py","attribute","Python attribute"],"3":["py","module","Python module"],"4":["py","function","Python function"],"5":["py","data","Python data"]},objtypes:{"0":"py:class","1":"py:method","2":"py:attribute","3":"py:module","4":"py:function","5":"py:data"},terms:{"140mb":3,"8gb":3,"abstract":[18,21,22,28,30,32,33,34,41],"boolean":[14,41],"class":[0,3,4,5,6,17,22,27,28,32,38],"default":[3,11,14,15,17,19,22,24,26,28,31,34,35,36,37,39,41,42],"export":11,"final":5,"float":[8,19,22,26,30,41,42],"function":[4,12,13,14,16,19,22,26,27,28,32,34],"import":[3,4,5,6,8,9,11,14,15,16,28,31,32],"int":[14,15,17,19,23,24,26,27,28,30,31,33,34,35,36,37,38,39,41,42],"k\u0131van\u00e7":26,"new":[0,6,8,11,14,19,26,32,41,42],"return":[14,16,17,19,22,26,28,30,31,33,34,35,36,37,38,39,41,42],"short":[8,12,32,39,41,42],"static":[14,28,30],"super":28,"switch":8,"true":[3,4,14,15,17,34,41,42],"try":[3,14],ABS:30,And:[0,2],For:[3,11,17,28,30,31],Ins:[3,4,6,8,15,16],The:[2,3,5,6,8,9,11,12,14,15,16,18,19,22,26,30,31,32,34],Then:0,There:8,These:[3,8,11,15,22,32],Use:11,Uses:[14,30,42],Using:[11,28],Will:[15,41,42],With:30,_________________________________________________________________:9,__call__:34,__init__:28,abc:[21,30,33,34],abl:2,about:[3,12],abov:[0,22,30],absolut:[5,30,31],absulot:31,academ:15,accept:[16,34,41],accuraci:[27,28],acm:20,activ:[0,11,22,28],activity_regular:22,actual:3,adam:28,add:[0,9,11,28,34],add_modifi:[4,34],add_testing_data:[9,28],add_training_data:[9,28],added:28,addit:[0,22],addition:3,after:[3,12,15,16,34],against:[30,31],ahead:2,alg:8,algorithm:[2,3,5,8,16,17,19,26,32,34],algorithm_numb:8,alist:17,all:[0,2,4,5,6,8,9,11,15,18,30,31,32],allow:8,allow_nan:17,along:[2,3,14,35,36,37,39],alreadi:11,also:[2,3,4,5,8,11,14,15,34],altern:30,alwai:8,amplitud:8,anaconda3:11,anaconda:[0,10],anconda:11,ani:[2,27,30],appli:[3,14,17,22,34,35,36,37,39],applic:[11,34],approach:[6,30,31,32],arang:5,aren:3,arg:[14,28],argument:[3,15,17,20,22,23,24,25,28,30,31,35,36,37,39,42],around:14,arrai:[4,9,14,17,30,33,34,38,41,42],arrang:8,aspect:12,assum:22,assur:4,atoi:17,attack:8,attempt:16,attribut:[27,28,30,32,34],audio:[2,3,4,5,6,8,9,12,14,15,16,19,20,26,28,30,31,34,35,36,37,39,41,42],audio_buff:[34,41,42],audio_fil:14,audio_fold:14,audio_folder_nam:15,audio_sampl:14,audiobuf:31,audiobuff:[4,5,6,12,13,16,19,26,30,31,34,35,36,37,39,41,42],auto:30,autom:12,automat:[0,8,11,14,15,19,23,24,25,26,27,28,30,31],avail:[0,2,8,11],axes:[14,33,34,38],axi:[14,33,34,38],band:[6,15,37],bandwidth:37,barkan:20,base:[3,4,6,15,16,18,19,20,21,22,23,24,25,26,27,28,30,31,32,33,34,35,36,37,39,41,42],base_lay:22,basic:[9,16,19],basicga:[6,16,18],batch:28,batch_siz:28,becaus:4,been:[8,16,22,28,34,41,42],befor:[14,32,34],being:[9,11,15,16,30],below:11,between:[5,6,8,19,22,26,28,30,31,41],bi_lstm:[4,5,9],bi_lstm_match:4,bi_lstm_prediction_:4,bia:22,bidirect:[9,23],bin:[5,30],binari:42,bool:[14,15,34,41,42],boost:11,both:[5,6],branch:0,browser:7,buffer:[14,28,30,31,35,36,39,41,42],buffer_s:41,build:[0,11,22],build_model:[20,23,24,25,28],built:[0,11,22],calcul:[6,19,26,28,30,31,33,34,35,36,38,39],call:[2,15,19,22,26,27,28,30,31,34,41],callback:[9,27,28],can:[0,2,3,4,8,9,11,12,14,15,16,17,22,28,30,32,41,42],cannot:11,captur:3,carri:5,categori:11,caus:[3,34],center:8,centroid:37,cepstral:[3,12,15,32,36],certain:[14,30],challeng:12,chang:[0,11],channel:[4,9,14],check:[2,11,31,34,42],check_circular:17,clamp_param:41,click:11,clip:15,clip_upp:30,clipper_upp:30,clone:[0,11],close:[2,3],cls:17,cnn:[4,7,20],cnn_extractor:4,cnn_matcher:4,cnn_prediction_:4,coars:8,code:[0,2,11],coeffici:[3,12,15,32,36],com:[11,22],come:11,command:[0,11],comment:0,compar:[2,30,31],comparison:32,compil:28,complex64:17,complex:[8,17,35,39],complex_dtyp:17,compon:36,compos:12,composit:12,comput:[15,22,23,24,25,38],compute_output_shap:22,conda:[0,11],config:22,configur:[2,3,15,20,22,41,42],connect:[9,22],construct:[15,20,23,24,25,26,28,30,31,34,35,36,37,39],constructor:[3,19,28],contain:[4,8,17,22,28,30,31,34,41],contrast:37,control:8,conv2d:9,conv2d_1:9,conv2d_2:9,conv2d_3:9,conv2d_4:9,conv2d_5:9,conv6:[9,18],convers:17,convert:17,convert_spectrum:17,convolut:[3,4,20],copi:0,core:[12,17,41,42],correct:[4,11,16,34],correspond:[4,28,41,42],could:[3,34],cov:0,coverag:0,creat:[0,3,4,8,9,11,14,15,22,28,30],creation:22,crossov:19,currect:15,current:[11,14,15,18,19,26,27,32,41,42],custom:22,cutoff:8,cxpb:19,data:[3,4,9,12,14,15,17,27,28,33,34,38],data_fm_mfcc:15,data_scal:[3,4,15],data_scaler_bas:38,data_simple_fm_mfcc:[3,4,9],data_simple_fm_stft:[3,4,9],datadir:0,datascalerbas:[32,34,38],dataset:[2,4,9,15,28,32,34],datasetgener:3,date:0,datesetgener:[12,13],dbraun:11,deactiv:11,deap:[0,11],debug:11,decai:8,deep:[2,3,5,12,15,20,23,24,25,28,30,31],def:28,defin:[26,28,32,34],definit:28,defualt:14,defuault:[37,39],delai:8,dens:[9,22,28,32],dense_1:9,dense_2:9,dense_3:9,dense_4:9,dense_5:9,dense_6:9,dense_7:9,depend:[11,30,35,36,37,39],depth:8,describ:30,descript:[8,41,42],design:[4,32],detail:[28,42],determin:[11,19,26,35,36,37,39],detun:8,dev:0,develop:[0,11,15,42],deviat:30,dex:[2,3,4,6,8,15,16],dexed_simple_fm:[3,4,6,8],dexed_simple_fm_init:8,dfine:30,dict:[16,27,30,41],dictionari:[22,27,30,31,41,42],differ:[2,8,15,30,31],differen:8,dimens:[22,32,34,35,36,37,39],dimension:24,directori:[0,14,15],disk:[6,14,16],displai:42,distanc:[30,31],distinguish:[30,31],distribut:[5,11],doan:30,doc:[0,15,22,28],docstr:0,document:[12,31],doe:[7,15,22],domin:26,don:11,dowload:2,download:[2,11],dropout:[9,22],dropout_1:9,dropout_2:9,dtype:17,dump:17,dure:[15,19,26,27,30,31,34,35,36,37,39,41],dx7:[2,8],dylib:11,dynamic_lookup:11,each:[3,4,5,6,9,15,17,19,26,27,30,31,34,36,37],earli:12,earlystop:9,easier:27,easiest:11,edg:30,edit:8,effect:[41,42],either:[17,41,42],electron:12,element:[7,17,19],emerg:[23,24,25],empti:14,emul:[2,8],enabl:11,encod:17,encourag:0,end:[4,14,27],engin:[11,22,41],ensure_ascii:17,entir:[14,28],env:[8,11],envelop:8,environ:[0,10],epoc:28,epoch:[9,27,28],error:[5,6,16,19,26,28,30,31],estim:[2,3,4,5,9,12,16,19,20,21,22,23,24,25,26,27,28,30,31],estimator_bas:[19,21,26,28],estimatorbas:[12,16,18,19,26,28],etc:30,euclidian:[30,31],euclidian_dist:30,eval_gener:3,evalu:[2,4,6,12,16,19,26,30,31,32],evaluate_target:[30,31],evaluation_bas:31,evaluation_stat:5,evaluationbas:[12,29,31],everi:6,evolutionari:12,exact:11,exampl:[2,3,14,15,16,28,30,31,32,34],except:[11,16],exist:[11,14],expect:[3,4,22],experi:[1,3,8,12,28],explor:12,extend:[8,28],extra:11,extraciton:36,extract:[3,4,9,15,16,17,19,26,31,34,36,37],extractor:[3,4,6,15,26],fals:[14,15,17,34,41],fast:[12,32,35],featur:[3,4,6,9,12,15,16,19,26,28,31,33,34,35,36,37,38,39],featurebas:16,features_bas:[35,36,37,39],features_filenam:15,features_list:26,featuresbas:[12,15,16,19,26,32,35,36,37,39],fed:[9,16],fedden:[11,23,24,25],feddon:42,feedback:8,fft:[6,12,32,34,36,37,39],fft_size:[3,4,35,39],fft_sze:35,figur:[12,14],file:[0,3,4,5,6,8,11,14,15,16,17,28,30,31,32,34,41,42],file_nam:15,file_prefix:[3,15],filenam:15,filepath:[16,28],filter:8,find:3,fine:8,finish:27,first:[0,3,8,11,15,16,17,30,31,42],fit:[6,9,19,26,28,32,33,34,38],fit_scal:34,fit_scaler_onli:15,fit_transform:33,fix:[8,11,28],flag:[8,11],flaot:[41,42],flat:37,flatten:[4,9,34,35,39],float32:17,focu:[8,12],folder:[4,6,9,14,15],folder_of_ga_predict:31,folder_of_mlp_predict:31,folder_of_target_audio_fil:31,follow:[0,2,11,18,32],forc:[30,41],fork:[0,11],form:42,format:[28,35,36,37,39],fourier:[12,32,35,39],frame:4,frame_s:[3,15,36,37],free:[2,22],freer:12,freez:8,frequenc:[3,8,12,15,32,36,37],from:[0,2,3,4,5,9,11,14,15,16,17,18,20,22,27,28,30,31,32,34,41,42],frozen:[41,42],fulli:[8,9],further:42,futur:[3,15,34],ga_extractor:[6,16],ga_match:[6,16],ga_predicition_:6,ga_predict:31,gain:8,gate:22,gener:[2,4,5,12,15,19,26,42],genet:[2,4,5,16,19,26],get:[0,4,12,41,42],get_audio:[14,41,42],get_config:22,get_featur:[34,35,36,37,39],get_paramet:[41,42],get_patch:[16,41,42],get_plotting_data:27,get_random_exampl:[41,42],get_sample_r:14,get_scor:30,get_stat:30,getter:14,git:[2,11],github:[2,11,22],given:[14,21,28],global:8,goal:2,going:8,googl:0,grayscal:4,greater:14,ground:[9,28,30,31],guid:12,handl:[4,14,22,32],has:[8,14,16,19,20,22,23,24,25,28,34,41,42],has_scal:34,have:[2,8,9,11,15,30,31,32,34,38,41,42],hdf5:28,header:11,hear:2,help:[4,32],helper:[6,27],her:[9,12],here:[0,2,3,9,11,22,28,34],hertz:14,hidden:24,high:18,higher:11,highwai:[22,23],highway_lay:[9,23,24],highway_layer_1:9,highway_layer_2:9,highway_layer_3:9,highway_layer_4:9,highway_layer_5:9,highwaylay:[9,18],hist:30,histogram:[30,31],hitogram:30,hold:32,hop:[36,37,39],hop_siz:[3,4,6,15,16,36,37,39],hour:6,how:[3,4,11,32],howev:11,html:[0,32],http:[11,22,32],human:[17,32],hwyblstm:[9,18],icon:11,ident:3,ieee:[20,23,24,25],iii:7,imag:[4,9],implement:[16,18,21,22,26,28,30,32,33,34],implment:38,includ:[11,22,30,41,42],incom:4,indent:17,independ:[34,36,37],index:[0,11,12,30,41,42],indic:[30,34,41,42],individu:[6,19,26],infom:27,inform:[11,22],inherit:[9,15,16,18,27,28,31,32,34],initi:[4,22],input:[4,9,14,16,17,19,20,21,22,23,24,25,26,28,30,31,32,34,35,36,37,39],input_list:[30,31],input_shap:[20,22,23,24,25,28],insid:11,inspir:32,instal:[2,10,12],instanc:[6,8,15,19,22,26,28,34],instanti:[9,14,22,31],instead:22,instruct:[0,2,11],integ:[4,17,22],integr:11,intellig:[12,23,24,25],interact:[11,42],interest:11,interfac:[18,32,34],inverno:[23,24,25],inversynth:20,ipython:0,is_valid_parameter_set:42,issu:11,istanti:28,its:22,john:[23,24,25],jorshi:11,journal:26,jshier:15,json:[3,4,5,6,8,17,30,31,41,42],juce:11,jucer:11,jump:2,just:[27,34],kadam:22,keep:17,kei:[8,17,27,30,41,42],kera:[9,18,22,27,28],keyword:[14,20,22,23,24,25,28,30,31,35,36,37,39,42],king:[23,24,25],know:3,known:12,kwarg:[14,20,22,23,24,25,28,30,31,35,36,37,39,42],label:28,lambda:[4,34],languag:20,larg:15,larger:35,last:41,later:[22,34,38],latter:8,lauri:12,layer:[20,22,23,24,25,28],lboost:11,learn:[2,3,5,12,15,24,25,28,30,31,32],leav:11,left:8,len:[4,6],length:[3,14,36,41,42],leon:[23,24,25,42],less:14,let:8,level:[8,18],lfo:8,lib:11,librari:[3,4,6,8,11,12,15,16,17,28],librenderman:[0,11,42],librendman:11,librosa:[0,11,14,32],like:[14,28,34],line:[0,11],linker:11,linux:11,list:[0,14,17,19,22,26,28,30,31,41,42],lit:[30,31],live:22,load:[3,4,5,6,8,9,14,16,28,34,41,42],load_fold:[4,5,6,14],load_model:28,load_patch:[41,42],load_plugin:42,load_scal:[4,34],load_stat:[3,4,6,41,42],load_weight:28,local:0,locat:[11,14,16,28,30,34,41,42],lock:41,log:27,log_data:27,logarithm:14,logger:[9,27],logic:22,longer:3,look:11,loss:[27,28],lpython3:11,lstm:[4,7,18,23],lstm_1:9,lstm_2:9,lstm_extractor:4,lstm_matcher:4,lstm_prediction_:4,lstm_size:23,mac:[0,11],machin:32,macosx:11,macret:26,made:[5,11,30],magnitu:17,magnitud:[3,4,6,17,35,39],magnitude_phas:[17,35,39],mai:[0,6],make:[0,2,11,17,28,30,31],manhattan:[30,31],manhattan_dist:30,manual:[8,11],map:[41,42],mark:[23,24,25],master:8,match:[1,3,5,12,16,22,28],match_from_fil:16,matplotlib:[0,5,11,14,27,30],matrix:[20,23,24,25,28,32,34],matthew:[23,24,25],matthieu:26,maximum:30,mean:[3,5,15,28,30,31,37,38],mean_abs_error:[5,30,31],mean_squared_error:30,meaning:32,measur:[5,19,26],median:30,mel:[3,12,15,32,36],member:34,method:[21,22,28,30,31,34,41],metric:[5,28,30,31],mfcc:[4,5,6,12,15,16,31,32],mfcc_eval_stat:31,mfcc_result:31,mfcceval:[5,12,29],middl:8,midi:[11,41],midi_not:[41,42],midi_veloc:[41,42],might:[11,30,31],minimum:30,miss:0,mlp:[4,7,9,18,28,31],mlp_extractor:4,mlp_matcher:4,mlp_predict:31,mlp_prediction_:4,mod:8,mode:[0,8],model:[2,3,15,18,20,22,23,24,25,27,28],modif:[11,34],modifi:[4,8,11,34,42],modul:[8,12,32],monitor:9,monophon:42,more:[12,32],most:[8,11,30],motiv:12,move:11,mpl:0,multi:[6,25,28],multipl:26,music:[12,26],must:[11,15,16,19,21,28,30,32,33,34,35,39,41,42],mutat:19,mutpb:19,name:[0,4,11,12,14,15,22,41,42],nativ:14,natur:[4,14,17],natural_kei:17,ndarrai:[14,17,28,30,33,34,35,36,37,38,39],need:[0,9,22,34],network:[3,4,20,22,23,24,25,28],neural:[3,20,28],newer:30,ngen:[6,19],nine:8,non:[9,17,26,41,42],none:[9,14,16,17,19,22,26,27,28,30,31,33,34,35,36,37,38,39,42],nor:22,normal:[14,15,32,34,38],note:[3,34,41],note_length_sec:[3,4,6,15,41,42],notebook:2,now:[4,6,8,9,11,34],npy:[3,9,15],nsga3:[6,18],nsga:7,nsga_extractor:6,nsga_match:6,nsga_prediction_:6,num_mfcc:[3,4,6,15,16,36],num_output:[20,23,24,25,28],num_sampl:14,number:[4,14,15,17,19,20,23,24,25,27,28,30,36,41,42],numpi:[0,5,9,11,14,15,17,30],numpynumberencod:17,object:[4,5,6,14,15,16,19,26,30,31,32,34],off:8,often:32,on_epoch_end:27,one:[3,4,6,9,15,19,22,26,31,32,34,35,39],onli:[8,11,15,19],onlin:2,open:[0,2,8,11,14,17],oper:[8,11],oppos:3,optim:28,option:[4,8,11,15,16,17,19,20,22,23,24,25,26,28,30,31,34,35,36,37,39,41,42],optiona:36,order:[2,3,4,8,11,16,17,34,41,42],oren:20,org:32,organ:27,orient:[3,34],origin:11,oscil:8,osx:11,other:[8,11,18,23,24,25,34],otherwis:[17,28,34],ouptut:3,our:[3,8,9,11],out:[2,5,11,12],outer:24,output:[3,4,6,8,9,15,16,17,20,22,23,24,25,28,34,35,36,37,39],output_fold:[3,15],outsid:34,over:37,overrid:[8,22,30,31,34,41,42],overridden:[6,8,41,42],overridden_param:41,overridden_paramet:8,overriden:4,packag:[0,11,42],pad:[14,35],page:[0,2,12],param:[4,6,8,9],param_rang:41,paramet:[2,3,8,9,14,15,16,17,19,20,21,22,23,24,25,26,27,28,30,31,33,34,35,36,37,38,39,41,42],parameter:42,parameter_index:42,parameter_valu:42,paramt:8,parikh:22,parikhkadam:22,part:2,pasquier:26,pass:[14,20,22,23,24,25,27,28,30,31,32,34,41,42],patch:[3,9,15,16,19,26,41,42],patches_filenam:15,path:[11,14,16,17,28,30,41,42],pathlib:14,patienc:9,peak:14,peak_norm:14,per:22,percept:32,perceptron:[25,28],perform:[2,4,5,6,16,32,33,38],phase:17,philipp:26,philosophi:12,pickl:[3,15,34],pioneer:12,pip:[0,10],pipelin:[4,34],pitch:8,pkl:[3,4,15],pleas:[0,11],plot:[5,9,14,27,30,31],plot_hist:[5,30,31],plot_spectrogram:14,plt:[5,14],plug:[3,4,6,8,15,16],plugin:[11,42],plugin_path:42,pop_siz:[6,19],popul:[6,19],portion:3,posit:[30,31],possibl:[2,3],potenti:34,power:[17,35,39],power_phas:[17,35,39],pre:[11,32,34],prection:[19,26],predict:[4,19,21,26,28,30,31],prediction_1:[30,31],prediction_1_for_target_1:[30,31],prediction_1_for_target_2:[30,31],prediction_2:[30,31],prediction_2_for_target_1:[30,31],prediction_2_for_target_2:[30,31],prefix:15,preprocess:32,prescal:34,preset:26,presetgen:26,previous:3,prior:[4,9,16,32,34],probabl:19,proce:11,process:[0,12,20,32,42],produc:5,program:[2,8,12,23,24,25,30,31],programmat:[8,11],project:[0,2,12],projuc:11,propos:[20,23,24,25],provid:[4,11,14,22,28,30,32,34,41,42],pull:[0,11],py36:11,pypi:11,pyplot:[5,14],pytest:0,python37:11,python3:11,python:[0,2,11,22,27],rais:16,random:[3,15,19,26,41,42],randomize_patch:[41,42],rang:[4,6,8,30,41],rate:[8,14,22,31,34,41],raw:[4,17,31,32,34],read:14,readi:0,recent:11,recommend:[0,11],recreat:2,refer:[0,11,34,41,42],regul:22,regular:[11,17,22],reinstanti:22,rel:9,releas:[3,8],relev:32,reli:42,reload:[3,8],relu:[22,28],remov:[3,11,15,41,42],renam:11,render:[3,15,19,26,41,42],render_length_sec:[3,4,6,15,41,42],render_patch:[41,42],rendered_patch:41,renderengin:11,renderman:[0,2,10],replac:[14,17],replace_audio_data:14,replic:[2,3],repo:[0,2,11],report:0,repositori:[0,11],repres:[19,26,30,31,32],represent:32,request:[0,11],requir:[4,10,12],resampl:14,research:26,reset:15,reshap:[4,9],resiz:14,reson:8,restructur:0,resul:16,result:[2,3,4,5,6,14,15,28,30,31,32,34,35,36,37,39,41,42],result_audio:16,result_patch:16,reus:8,rice:30,right:30,rms_error:28,rolloff:37,root:[0,28],rst:0,run:[0,2,3,4,6,11,16,19,26,28,30,31,34,35,36,39],sacl:33,same:[4,17,22,30,31],sampl:[4,6,14,15,19,26,30,31,34,36,37,39,41],sample_r:[14,31,34,41],sandbox:11,save:[3,4,5,6,8,9,14,15,17,28,30,31,34,41,42],save_audio:[3,15],save_model:[9,28],save_scal:[3,15,34],save_scores_json:[30,31],save_st:[8,41,42],save_stats_json:[5,30,31],save_weight:28,saved_model:[4,9],savedmodel:28,scale:[3,4,8,12,15,33,34,35,36,37,38,39],scale_axi:[34,35,36,37,39],scaler:[4,15,32,33,34,38],scikit:32,score:30,scott:30,search:[11,12],second:[3,8,14,15,17,30,41],section:11,see:[0,2,11,14,22,28,31,32,34,35,36,37,39,42],seed:[19,26],select:[2,8,17],self:28,sensit:8,sent:11,separ:[3,15,17],sequenc:30,sequenti:[9,28],sequential_1:9,sequential_2:9,sequential_3:9,seri:[4,36],serializ:22,serv:12,set:[3,4,6,8,9,14,15,19,26,28,30,34,35,36,37,39,41,42],set_overridden_paramet:[8,41,42],set_patch:[41,42],set_scal:34,setup:[0,3,4,6,8,9,15,16],sever:[2,6],shape:[4,9,17,20,22,23,24,25,28,41,42],she:12,shift:[37,39],should:[21,27,28,30,31,34,41],show:[5,8,11,14],shuffl:28,shuffle_s:28,signal:[3,14,20],simpl:[8,25,28],simple_fm_bi_lstm:[4,9],simple_fm_cnn:[4,9],simple_fm_lstm:[4,9],simple_fm_mlp:[4,9],simplifi:[8,32],sinc:3,sine:8,singl:[4,6,19,28,30],site:11,six:8,size:[3,14,15,19,28,35,36,37,39,41],skip_overridden:[41,42],skipkei:17,sklearn:[32,34],slice:[4,9,34],small:[3,8,11],smaller:35,some:[4,8,21,34],some_audio:34,some_audio_fil:14,sort:[4,14,17,26],sort_kei:17,sound:[1,3,5,8,12,16,19,23,24,25,26,28],soundmatch:[4,6,12,13],sourc:[0,2,30,31],source_:30,space:24,specif:[4,11,14,41,42],specifi:[4,11],specshow:14,spectral:[6,12,32,37],spectralsummar:[6,12,32],spectrogram:14,spectrum:[17,34],spectrum_typ:17,speech:20,speed:8,speigel:11,spgl:[3,4,5,6,8,9,14,15,16,31],sphinx:0,sphinx_rtd_them:0,spiegel:[12,15],spiegelib:[0,2,3,4,5,6,8,9,11,14,15,16,17,19,20,21,22,23,24,25,26,27,28,30,31,32,33,34,35,36,37,38,39,41,42],spiegelib_dev:0,spiegelib_env:11,sqrt:30,squar:[28,30,31],src:0,stabl:32,stackoverflow:17,standar:38,standard:[3,15,30,32],standardscal:32,start:[8,12,14],start_sampl:14,stat:[30,31],state:[22,41,42],statist:30,std:38,step:[22,32],stft:[4,9,12,32],still:11,store:[14,15,27,28,30],str:[14,15,16,17,28,30,34,35,39,41,42],strategi:30,string:[15,17,28,30],sturg:30,sub:11,subclass:22,submit:0,subset:8,suffix:28,summar:[5,6,12,30,32,37],summari:9,support:[7,11],sure:[0,2,11,17,30,31],sustain:8,sync:8,synth:[3,4,6,8,9,12,15,16,19,26,41,42],synth_bas:42,synth_param:[3,4,6,8],synthbas:[12,15,16,19,26,40,42],synthes:[2,3,4,11,12,15,16,19,20,21,23,24,25,26,28,30,31,41,42],synthvst:[3,4,6,8,12,15,16,40],system:[11,19,26],take:6,target:[2,3,4,5,6,7,16,19,26,28,30,31],target_1:[30,31],target_2:[30,31],tatar:26,techniqu:[23,24,25],technolog:12,tensflow:28,tensor:[22,28],tensorflow:[0,3,9,11,18,22,27,28],tensorshap:22,term:0,test:[3,5,11,12,15,34],test_:[3,15],test_data:28,test_featur:9,test_patch:9,testfeatur:9,testparam:9,text:[0,17],tf_estimator_bas:[20,23,24,25],tfepochlogg:[9,18],tfestimatorbas:[4,9,18,20,23,24,25],than:[3,14,35],thei:[11,15,22,32],them:15,thi:[0,2,3,4,5,6,8,11,12,14,15,16,19,22,26,27,28,30,31,32,34,38,41,42],those:[12,19,26],through:[2,8,11,34],throughout:8,time:[4,9,11,12,19,26,32,34,36,37,39],time_major:[3,4,15,34],time_slic:[3,34],topic:[23,24,25],total:[3,9,41,42],tox:0,tqdm:[0,11],train:[2,3,4,15,22,27,28,32,34],train_:[3,15],train_data:28,train_featur:9,train_patch:9,trainabl:9,trainfeatur:9,trainparam:9,transact:[20,23,24,25],transform:[12,22,32,33,34,35,38,39],transform_dropout:22,transform_gate_bia:22,treat:[8,34],tree:11,trim:14,truncat:35,truth:[9,28,30,31],tune:8,tupl:[20,22,23,24,25,28,33,34,35,36,37,38,39,41,42],turn:8,twine:0,two:[6,8,30,31],type:[4,9,14,15,16,17,19,26,27,28,30,31,33,34,35,36,37,38,39,41,42],typeerror:16,typic:[22,28],unaffect:42,undefin:11,under:11,uneffect:41,unit:[0,3,15],until:11,updat:[0,8,11,41,42],update_midi_buff:11,upon:28,upper:30,usag:17,use:[0,2,4,8,11,15,16,17,19,26,30,31,32,33,34,35,36,38],used:[0,2,3,4,5,6,9,11,15,17,19,22,26,30,31,32,34,41,42],useful:15,user:[11,15],usernam:11,uses:[4,6,11,14,30,34],using:[0,2,3,4,6,9,11,12,16,19,23,24,25,26,27,28,30,31,34,37],util:[12,13,14,18],val_loss:9,valid:[3,9,15,27,28,42],valu:[3,8,9,15,17,19,26,28,30,34,41,42],variabl:[22,34,35,39,41],varianc:[3,15,37,38],variou:[8,12],vector:[15,17,22,37],veloc:[8,41],verif:[30,31],verifi:[11,30],verify_audio_input_list:30,verify_input_list:[30,31],version:[0,11],view:0,virtual:11,visit:12,vst:[2,3,4,6,8,11,15,16,23,24,25,42],wai:[8,11,17],want:[2,3,11,30,31],wav:[3,4,6,14,16,34],wave:8,waveform:8,websit:12,weight:[22,28],weights_path:28,welcom:0,well:14,were:[4,11,30],whatev:[11,12],when:[0,4,11,27,28,35,36,37,39],where:[11,22,30,31,34,41,42],whether:[15,34,41,42],which:[3,5,6,8,9,11,12,16,28,30,32,34,35,36,37,39],window:11,within:[11,15,32],without:22,work:[3,11,12,15],worri:3,would:[30,31],wrap:[11,28],wrapper:32,write:[0,11,14],written:0,xcode:11,y_pred:28,y_true:28,yamaha:[2,8],yee:[23,24,25],yet:41,you:[0,2,7,11,30,31],your:[0,11],zero:[14,35,38],zsh:0},titles:["Contribution Guide","Examples","FM Sound Match Experiment","Dataset Generation","Sound Match Deep Learning Models","Evaluation","Sound Match Genetic Algorithm Estimators","Sound Matching Audio Clips","Synthesizer Configuration","Train Deep Learning Models","Getting Started","Installation","SpiegeLib","SpiegeLib Core Classes","AudioBuffer Class","DatasetGenerator Class","SoundMatch Class","Utility Functions","Estimator Classes","BasicGA Class","Conv6 Class","EstimatorBase Class","HighwayLayer","HwyBLSTM Class","LSTM Class","MLP Class","NSGA3 Class","TFEpochLogger","TFEstimatorBase","Evaluation Classes","EvaluationBase Class","MFCCEval Class","Audio Feature Extraction","DataScalerBase Class","FeaturesBase Class","FFT Class","MFCC Class","SpectralSummarized Class","StandardScaler Base","STFT Class","Synth Classes","SynthBase Class","SynthVST Class"],titleterms:{"class":[13,14,15,16,18,19,20,21,23,24,25,26,29,30,31,33,34,35,36,37,39,40,41,42],"function":17,"long":9,"short":9,algorithm:6,anaconda:11,audio:[7,32],audiobuff:14,base:38,basic:6,basicga:19,clip:7,cnn:[5,9],configur:8,contribut:[0,12],conv6:20,convolut:9,core:13,data:32,datascalerbas:33,dataset:3,datasetgener:15,deep:[4,9,18],direct:9,document:0,domin:6,environ:11,estim:[6,18],estimatorbas:21,evalu:[3,5,29],evaluationbas:30,evolutionari:18,exampl:[1,12],experi:2,extract:32,featur:32,featuresbas:34,fft:35,first:12,gener:[0,3],genet:6,get:10,guid:0,highwai:9,highwaylay:22,histogram:5,hwyblstm:23,iii:[5,6],indic:12,instal:[0,11],layer:9,learn:[4,9,18],lstm:[5,9,24],match:[2,4,6,7],memori:9,mfcc:[3,36],mfcceval:31,mlp:[5,25],model:[4,9],multi:9,network:9,neural:9,non:6,nsga3:26,nsga:[5,6],perceptron:9,pip:11,refer:12,renderman:11,requir:[0,11],scale:32,section:2,sort:6,sound:[2,4,6,7],soundmatch:16,spectralsummar:37,spiegelib:[12,13],standardscal:38,start:10,step:12,stft:[3,39],synth:40,synthbas:41,synthes:8,synthvst:42,tabl:12,term:9,test:0,tfepochlogg:27,tfestimatorbas:28,train:9,util:17}})