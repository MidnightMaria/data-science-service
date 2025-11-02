package com.agnesmaria.datascience.service;

import com.agnesmaria.datascience.config.IntegrationConfig;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Service;
import org.springframework.web.client.RestTemplate;

@Service
public class DataFetchService {

    private static final Logger log = LoggerFactory.getLogger(DataFetchService.class);

    private final RestTemplate restTemplate;
    private final IntegrationConfig integrationConfig;

    public DataFetchService(RestTemplate restTemplate, IntegrationConfig integrationConfig) {
        this.restTemplate = restTemplate;
        this.integrationConfig = integrationConfig;
    }

    public String fetchInventoryData() {
        String url = integrationConfig.getInventoryService().getUrl();
        log.info("Fetching inventory data from {}", url);
        return restTemplate.getForObject(url, String.class);
    }

    public String fetchRetailData() {
        String url = integrationConfig.getRetailService().getUrl();
        log.info("Fetching retail data from {}", url);
        return restTemplate.getForObject(url, String.class);
    }

    public String fetchAll() {
        log.info("Fetching all data from inventory & retail");
        String inventory = fetchInventoryData();
        String retail = fetchRetailData();
        return String.format("{\"inventory\": %s, \"retail\": %s}", inventory, retail);
    }
}
