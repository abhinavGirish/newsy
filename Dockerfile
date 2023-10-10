# Use an official Python 3.8 runtime as a base image
FROM python:3.8

# Set environment variables
# ensures that Python outputs everything that's printed directly to the terminal (so logs can be seen in real-time)
ENV PYTHONUNBUFFERED=TRUE
# ensures Python doesn't try to write .pyc files to disk (useful for improving performance in some scenarios)
ENV PYTHONDONTWRITEBYTECODE=TRUE
# Update PATH environment variable to include /opt/program directory
ENV PATH="/opt/program:${PATH}"

# Set the working directory in the Docker image to /opt/program
WORKDIR /opt/program

# Upgrade pip package manager to the latest version
RUN pip install --upgrade pip

# Get requirements.txt file and install packages
COPY ./requirements.txt /opt/program/requirements.txt
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Copy the source code of the application
COPY ./src /opt/program

#ENTRYPOINT ["gunicorn", "--bind", "unix:/tmp/gunicorn.sock", "-b", ":8080", "prediction_script:app"]
ENTRYPOINT ["gunicorn", "-b", "unix:/tmp/gunicorn.sock", "prediction_script:app"]