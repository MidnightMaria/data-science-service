package com.agnesmaria.datascience;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.context.annotation.Bean;
import org.springframework.web.client.RestTemplate;

@SpringBootApplication
public class DataScienceServiceApplication {

    public static void main(String[] args) {
        SpringApplication.run(DataScienceServiceApplication.class, args);
    }

    // ✅ Tambahkan ini untuk mendaftarkan RestTemplate ke Spring context
    @Bean
    public RestTemplate restTemplate() {
        return new RestTemplate();
    }
}
