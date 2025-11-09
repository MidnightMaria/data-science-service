# ==========================================
# 🧠 Data Science Service Dockerfile
# ==========================================

# Stage 1: Build the JAR
FROM maven:3.9.9-eclipse-temurin-17 AS build
WORKDIR /app

COPY pom.xml .
RUN mvn dependency:resolve -B || true

COPY src ./src
RUN mvn clean package -DskipTests

# Stage 2: Run the app
FROM eclipse-temurin:17-jre-jammy
WORKDIR /app

# 🔧 Pastikan semua folder dataset & processed dibuat dan writable
RUN mkdir -p /app/data/processed /app/data/raw /app/dataset/raw && chmod -R 777 /app/data

COPY --from=build /app/target/*.jar app.jar

EXPOSE 8084
ENTRYPOINT ["java", "-jar", "app.jar"]
