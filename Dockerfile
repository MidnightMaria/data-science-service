# ==========================================
# 🧠 Data Science Service Dockerfile (final)
# ==========================================

# Stage 1: Build Spring Boot JAR
FROM maven:3.9.9-eclipse-temurin-17 AS build
WORKDIR /app
COPY pom.xml .
RUN mvn dependency:resolve -B || true
COPY src ./src
RUN mvn clean package -DskipTests

# Stage 2: Run the app + Streamlit dashboard
FROM eclipse-temurin:17-jre-jammy
WORKDIR /app

# Copy JAR dari stage 1
COPY --from=build /app/target/*.jar app.jar

# Install Python + Streamlit + dependencies
COPY requirements.txt .
RUN apt-get update && apt-get install -y python3 python3-pip && \
    pip install --no-cache-dir -r requirements.txt && \
    apt-get clean

# Copy dashboard Streamlit dan folder data
COPY reports /app/reports
COPY data /app/data

# Buat folder untuk output / logs (jika belum ada)
RUN mkdir -p /app/data/processed /app/logs

# Expose port untuk Spring Boot (8084) dan Streamlit (8501)
EXPOSE 8084 8501

# Jalankan Spring Boot + Streamlit secara bersamaan
CMD java -jar app.jar & \
    streamlit run /app/reports/dashboard_inventory_optimization.py --server.port 8501 --server.address 0.0.0.0
