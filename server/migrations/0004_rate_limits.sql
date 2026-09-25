CREATE TABLE IF NOT EXISTS push_rate_limits (
    subject_hash BINARY(32) NOT NULL,
    window_started_at DATETIME(6) NOT NULL,
    request_count INT UNSIGNED NOT NULL DEFAULT 0,
    updated_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6) ON UPDATE CURRENT_TIMESTAMP(6),
    PRIMARY KEY (subject_hash, window_started_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

SET @schema_name = DATABASE();
SET @statement = IF((SELECT COUNT(*) FROM information_schema.statistics WHERE table_schema = @schema_name AND table_name = 'push_rate_limits' AND index_name = 'idx_push_rate_limits_cleanup') = 0, 'CREATE INDEX idx_push_rate_limits_cleanup ON push_rate_limits (updated_at)', 'SELECT 1');
PREPARE migration_statement FROM @statement;
EXECUTE migration_statement;
DEALLOCATE PREPARE migration_statement;
