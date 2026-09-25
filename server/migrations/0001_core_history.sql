-- Compatible creation and augmentation of location_history.
CREATE TABLE IF NOT EXISTS location_history (
    id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
    location_id INT NOT NULL,
    is_closed TINYINT(1) NULL,
    current_capacity INT NULL,
    max_capacity INT NULL,
    last_updated DATETIME(6) NULL,
    source_updated_at DATETIME(6) NULL,
    fetched_at DATETIME(6) NULL,
    PRIMARY KEY (id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

SET @schema_name = DATABASE();
SET @statement = IF((SELECT COUNT(*) FROM information_schema.columns WHERE table_schema = @schema_name AND table_name = 'location_history' AND column_name = 'is_closed') = 0, 'ALTER TABLE location_history ADD COLUMN is_closed TINYINT(1) NULL', 'SELECT 1');
PREPARE migration_statement FROM @statement;
EXECUTE migration_statement;
DEALLOCATE PREPARE migration_statement;

SET @statement = IF((SELECT COUNT(*) FROM information_schema.columns WHERE table_schema = @schema_name AND table_name = 'location_history' AND column_name = 'current_capacity') = 0, 'ALTER TABLE location_history ADD COLUMN current_capacity INT NULL', 'SELECT 1');
PREPARE migration_statement FROM @statement;
EXECUTE migration_statement;
DEALLOCATE PREPARE migration_statement;

SET @statement = IF((SELECT COUNT(*) FROM information_schema.columns WHERE table_schema = @schema_name AND table_name = 'location_history' AND column_name = 'max_capacity') = 0, 'ALTER TABLE location_history ADD COLUMN max_capacity INT NULL', 'SELECT 1');
PREPARE migration_statement FROM @statement;
EXECUTE migration_statement;
DEALLOCATE PREPARE migration_statement;

SET @statement = IF((SELECT COUNT(*) FROM information_schema.columns WHERE table_schema = @schema_name AND table_name = 'location_history' AND column_name = 'last_updated') = 0, 'ALTER TABLE location_history ADD COLUMN last_updated DATETIME(6) NULL', 'SELECT 1');
PREPARE migration_statement FROM @statement;
EXECUTE migration_statement;
DEALLOCATE PREPARE migration_statement;

SET @statement = IF((SELECT COUNT(*) FROM information_schema.columns WHERE table_schema = @schema_name AND table_name = 'location_history' AND column_name = 'source_updated_at') = 0, 'ALTER TABLE location_history ADD COLUMN source_updated_at DATETIME(6) NULL', 'SELECT 1');
PREPARE migration_statement FROM @statement;
EXECUTE migration_statement;
DEALLOCATE PREPARE migration_statement;

UPDATE location_history
SET source_updated_at = last_updated
WHERE source_updated_at IS NULL AND last_updated IS NOT NULL;

SET @statement = IF((SELECT COUNT(*) FROM information_schema.columns WHERE table_schema = @schema_name AND table_name = 'location_history' AND column_name = 'fetched_at') = 0, 'ALTER TABLE location_history ADD COLUMN fetched_at DATETIME(6) NULL', 'SELECT 1');
PREPARE migration_statement FROM @statement;
EXECUTE migration_statement;
DEALLOCATE PREPARE migration_statement;

SET @statement = IF((SELECT COUNT(*) FROM information_schema.statistics WHERE table_schema = @schema_name AND table_name = 'location_history' AND index_name = 'idx_location_history_location_fetched') = 0, 'CREATE INDEX idx_location_history_location_fetched ON location_history (location_id, fetched_at)', 'SELECT 1');
PREPARE migration_statement FROM @statement;
EXECUTE migration_statement;
DEALLOCATE PREPARE migration_statement;

SET @statement = IF((SELECT COUNT(*) FROM information_schema.statistics WHERE table_schema = @schema_name AND table_name = 'location_history' AND index_name = 'idx_location_history_location_source_updated') = 0, 'CREATE INDEX idx_location_history_location_source_updated ON location_history (location_id, source_updated_at)', 'SELECT 1');
PREPARE migration_statement FROM @statement;
EXECUTE migration_statement;
DEALLOCATE PREPARE migration_statement;
