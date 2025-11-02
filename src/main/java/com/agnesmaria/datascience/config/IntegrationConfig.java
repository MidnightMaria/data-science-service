package com.agnesmaria.datascience.config;

import org.springframework.boot.context.properties.ConfigurationProperties;
import org.springframework.context.annotation.Configuration;

@Configuration
@ConfigurationProperties(prefix = "integration")
public class IntegrationConfig {

    private ServiceConfig inventoryService;
    private ServiceConfig retailService;

    // ✅ Getter & Setter manual agar tetap aman walau tanpa Lombok
    public ServiceConfig getInventoryService() {
        return inventoryService;
    }

    public void setInventoryService(ServiceConfig inventoryService) {
        this.inventoryService = inventoryService;
    }

    public ServiceConfig getRetailService() {
        return retailService;
    }

    public void setRetailService(ServiceConfig retailService) {
        this.retailService = retailService;
    }

    public static class ServiceConfig {
        private String url;

        public String getUrl() {
            return url;
        }

        public void setUrl(String url) {
            this.url = url;
        }
    }
}
