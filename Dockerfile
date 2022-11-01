# Pull SDK image as base image
FROM rasa/rasa-sdk:2.6.0

# Change to root user to install dependencies
USER root

# Use subdirectory as working directory
WORKDIR /app

# Copy actions requirements
COPY actions/requirements-actions.txt ./

# Install extra requirements for actions code
RUN apt-get update && apt-get install build-essential -y
RUN apt-get install -y sqlite3 libsqlite3-dev
RUN pip install -r requirements-actions.txt

# Copy actions code to working directory
COPY ./actions /app/actions
COPY ./init_db.py /app/
COPY ./fbplots /app/fbplots
COPY ./supp_materials/H1_train_texts.csv /app/supp_materials/
COPY ./supp_materials/scenarios.txt /app/supp_materials/
COPY ./supp_materials/trs_per_depth.txt /app/supp_materials/
COPY ./supp_materials/per_schema_models /app/supp_materials/per_schema_models

# Don't use root user to run code
# USER 1001