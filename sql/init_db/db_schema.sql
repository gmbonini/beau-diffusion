CREATE TABLE IF NOT EXISTS feedback (
    id INT AUTO_INCREMENT PRIMARY KEY,
    step ENUM('IMAGE', 'VIDEO', 'REGENERATE') NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    is_positive BOOLEAN NOT NULL,
    original_prompt TEXT NOT NULL,
    refined_prompt TEXT NOT NULL,
    chat TEXT,
    negative_prompt TEXT,
    video_frame_url VARCHAR(255)
);

CREATE TABLE IF NOT EXISTS multiviews (
    id INT AUTO_INCREMENT PRIMARY KEY,
    feedback_id INT NOT NULL,
    image_url VARCHAR(255) NOT NULL,
    FOREIGN KEY (feedback_id) REFERENCES feedback(id) ON DELETE CASCADE
);
