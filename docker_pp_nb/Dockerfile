# Iguanas docker image (for PP NBs)
# If getting ERROR: THESE PACKAGES DO NOT MATCH THE HASHES FROM THE 
# REQUIREMENTS FILE, just keep re-running

# Install default environment
FROM dockerhub.paypalcorp.com/core-data-platform/ppmagics-spark230:latest
# FROM dockerhub.paypalcorp.com/core-data-platform/ppmagics-spark232:0.4.0 
ENV LANG C.UTF-8

# Switch to root user
USER root

# Copy Iguanas directory
RUN mkdir /Iguanas
COPY ./ /Iguanas

# Set certificates
RUN pip config set global.cert /Iguanas/combined_cacerts.pem
# RUN mkdir /cert
# COPY ./combined_cacerts.pem /cert
# RUN pip config set global.cert /cert/combined_cacerts.pem

# Upgrade setuptools and pip
RUN pip install --upgrade pip --no-cache-dir
RUN pip install --upgrade setuptools --no-cache-dir

# RUN pip install -r /iguanas/requirements.txt --no-cache-dir
# RUN pip install category-encoders==2.0.0 --no-cache-dir
# RUN pip install matplotlib==3.0.3 --no-cache-dir
# RUN pip install seaborn==0.9.0 --no-cache-dir
# RUN pip install numpy==1.19.4 --no-cache-dir
# RUN pip install pandas==1.1.4 --no-cache-dir
# RUN pip install hyperopt==0.2.5 --no-cache-dir
# RUN pip install joblib==0.16.0 --no-cache-dir
# RUN pip install pytest==6.0.1 --no-cache-dir
# RUN pip install requests==2.20.1 --no-cache-dir
# RUN pip install httpretty==1.0.5 --no-cache-dir
# RUN pip install scikit-learn==0.23.2 --no-cache-dir
# RUN pip install koalas==1.8.1 --no-cache-dir

# # Install sub-packages
# RUN pip install --no-deps /iguanas/iguanas/correlation_reduction/. --no-cache-dir
# RUN pip install --no-deps /iguanas/iguanas/rbs/. --no-cache-dir
# RUN pip install --no-deps /iguanas/iguanas/rule_application/. --no-cache-dir
# RUN pip install --no-deps /iguanas/iguanas/rule_generation/. --no-cache-dir
# RUN pip install --no-deps /iguanas/iguanas/rule_optimisation/. --no-cache-dir
# RUN pip install --no-deps /iguanas/iguanas/rule_scoring/. --no-cache-dir
# RUN pip install --no-deps /iguanas/iguanas/rule_selection/. --no-cache-dir
# RUN pip install --no-deps /iguanas/iguanas/rules/. --no-cache-dir
# RUN pip install --no-deps /iguanas/iguanas/utils/. --no-cache-dir

# # Install Iguanas
# RUN pip install --no-deps /iguanas/. --no-cache-dir
# RUN pip install pyspark==3.1.2 --user --no-cache-dir

# RUN pip install category-encoders==2.0.0 --no-cache-dir
# RUN pip install matplotlib==3.0.3 --no-cache-dir
# RUN pip install seaborn==0.9.0 --no-cache-dir
# RUN pip install numpy==1.19.4 --no-cache-dir
# RUN pip install pandas==1.1.4 --no-cache-dir
# RUN pip install hyperopt==0.2.5 --no-cache-dir
# RUN pip install joblib==0.16.0 --no-cache-dir
# RUN pip install scikit-learn==0.23.2 --no-cache-dir
# RUN pip install scipy==1.* --no-cache-dir

# RUN pip install --no-deps iguanas==0.0.1
# RUN pip install --no-deps iguanas[spark]==0.0.1
# RUN pip install iguanas==0.0.1
RUN pip install /Iguanas/.
RUN pip install /Iguanas/.[dev]
RUN pip install /Iguanas/.[spark]
# RUN pip install koalas==1.8.1 
# RUN pip install pyspark==3.1.2

RUN rm /Iguanas/combined_cacerts.pem