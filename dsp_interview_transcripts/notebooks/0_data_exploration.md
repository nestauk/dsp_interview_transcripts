# Notes on transcript data

Audio transcripts
Different languages - need translating
Typos (heating > eating)
Removing boiler plate messages and responses


## Schema

| Column Name      | Description                                                                            |
|------------------|----------------------------------------------------------------------------------------|
| `uuid`           | Unique identifier for each record or conversation entry.                               |
| `created_at`     | The date and time when the entry was created.                                           |
| `timestamp`      | The exact time the event or message was logged during the conversation.                 |
| `conversation`   | The unique identifier to the entire conversation thread.                  |
| `role`           | The role of the participant in the conversation (e.g., user, or bot)            |
| `is_hidden`      | Boolean flag indicating whether the entry is hidden or not.                             |
| `is_pending`     | Boolean flag indicating whether the entry is pending approval or processing.            |
| `text`           | The text content of the message or conversation.                                        |
| `audio`          | Audio data associated with the conversation entry, if any.                              |
| `audio_url`      | URL link to the audio file associated with the conversation.                            |
| `transcript`     | Text transcript of the audio content, if available.                                  |
| `image`          | Image data associated with the conversation entry, if any.                              |
| `image_url`      | URL link to the image associated with the conversation.                                 |
| `caption`        | A caption or description for the associated image or media.                             |