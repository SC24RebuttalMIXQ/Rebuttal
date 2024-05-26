
# pip install rouge_score
# pip install nltk
# rm -r /usr/local/lib/python3.10/dist-packages/tensorrt_llm
export PYTHONPATH=$PYTHONPATH:/code/tensorrt_llm/manual_plugin
export PYTHONPATH=$PYTHONPATH:/code/tensorrt_llm/manual_plugin/AutoAWQ
export FT_LOG_LEVEL=ERROR
cd tensorrt_llm
mv /usr/local/lib/python3.10/dist-packages/tensorrt_llm/libs libs
cd ..
# pip install modelutils -i https://pypi.tuna.tsinghua.edu.cn/simple
# pip install peft -i https://pypi.tuna.tsinghua.edu.cn/simple
cd quantkernel
python setup.py install
cd ..

# cd AutoAWQ/AutoAWQ_kernels-0.0.6
# python setup.py install
# cd ..
# cd ..


cd EETQ
python setup.py install
cd ..


cd build
cmake ..
make
cd ..