# Use Python 3.11.4 base image
FROM python:3.11.4

# Set the working directory to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --default-timeout=10000 -r requirements.txt


# Copy the 'templates' folder into the container
COPY templates/ templates/

# Make port 5000 available to the world outside this container
EXPOSE 3000

# Define environment variable (optional)
ENV NAME World

# Run app.py when the container launches
CMD ["python", "app.py"]
