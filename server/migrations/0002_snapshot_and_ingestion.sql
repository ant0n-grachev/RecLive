CREATE TABLE IF NOT EXISTS location_snapshot (
    location_id INT NOT NULL,
    is_closed TINYINT(1) NULL,
    current_capacity INT NULL,
    max_capacity INT NULL,
    source_updated_at DATETIME(6) NULL,
    fetched_at DATETIME(6) NOT NULL,
    created_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
    updated_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6) ON UPDATE CURRENT_TIMESTAMP(6),
    PRIMARY KEY (location_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

CREATE TABLE IF NOT EXISTS ingestion_runs (
    id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
    started_at DATETIME(6) NOT NULL,
    completed_at DATETIME(6) NULL,
    status VARCHAR(16) NOT NULL,
    received_count INT UNSIGNED NOT NULL DEFAULT 0,
    valid_count INT UNSIGNED NOT NULL DEFAULT 0,
    history_inserted_count INT UNSIGNED NOT NULL DEFAULT 0,
    snapshot_updated_count INT UNSIGNED NOT NULL DEFAULT 0,
    observed_location_ids JSON NOT NULL DEFAULT (JSON_ARRAY()),
    error_category VARCHAR(64) NULL,
    error_message VARCHAR(240) NULL,
    PRIMARY KEY (id),
    CONSTRAINT chk_ingestion_runs_status CHECK (status IN ('running', 'succeeded', 'failed'))
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

SET @schema_name = DATABASE();
SET @statement = IF((SELECT COUNT(*) FROM information_schema.statistics WHERE table_schema = @schema_name AND table_name = 'location_snapshot' AND index_name = 'idx_location_snapshot_fetched_at') = 0, 'CREATE INDEX idx_location_snapshot_fetched_at ON location_snapshot (fetched_at)', 'SELECT 1');
PREPARE migration_statement FROM @statement;
EXECUTE migration_statement;
DEALLOCATE PREPARE migration_statement;

SET @statement = IF((SELECT COUNT(*) FROM information_schema.statistics WHERE table_schema = @schema_name AND table_name = 'ingestion_runs' AND index_name = 'idx_ingestion_runs_status_completed') = 0, 'CREATE INDEX idx_ingestion_runs_status_completed ON ingestion_runs (status, completed_at)', 'SELECT 1');
PREPARE migration_statement FROM @statement;
EXECUTE migration_statement;
DEALLOCATE PREPARE migration_statement;
