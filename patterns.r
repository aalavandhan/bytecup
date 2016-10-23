require(data.table)
require(plyr)

user_info <- read.table("data/user_info.txt", sep="\t", colClasses = "character") 
names(user_info) <- c("user_id", "expert_tags", "word_id", "char_id")

question_info <- read.table("data/question_info.txt", sep="\t", colClasses = "character") 
names(question_info) <- c("question_id", "tag", "word_id", "char_id", "upvotes", "answers", "top_answers")
question_info$tag <- as.integer(question_info$tag)
question_info$top_answers <- as.numeric(question_info$top_answers)
question_info$answers <- as.numeric(question_info$answers)

answer_info <- read.table("data/invited_info_train.txt", sep="\t", colClasses = "character") 
names(answer_info) <- c("question_id", "user_id", "answered")

splitter <- strsplit(user_info$expert_tags, split = "/")
user_expert_info <- data.frame(user_id = rep(user_info$user_id, sapply(splitter, length)), expert_tags = unlist(splitter))
user_expert_info$expert_tags = as.integer(user_expert_info$expert_tags)
user_expert_info$user_id = as.character(user_expert_info$user_id)

splitter <- strsplit(user_info$word_id, split = "/")
user_desc_word <- data.frame(user_id = rep(user_info$user_id, sapply(splitter, length)), word_id = unlist(splitter))

splitter <- strsplit(user_info$char_id, split = "/")
user_desc_char <- data.frame(user_id = rep(user_info$user_id, sapply(splitter, length)), char_id = unlist(splitter))

splitter <- strsplit(question_info$word_id, split = "/")
question_desc_word <- data.frame(user_id = rep(question_info$question_id, sapply(splitter, length)), word_id = unlist(splitter))

splitter <- strsplit(question_info$char_id, split = "/")
question_desc_char <- data.frame(user_id = rep(question_info$question_id, sapply(splitter, length)), char_id = unlist(splitter))


# Number of expert users : 28762
length(unique(user_info$user_id))

# Number of questions : 8094
length(unique(question_info$question_id))

# Number of questions answered : 27324
nrow(answer_info[answer_info$answered == 1, ])

# Ratio questions answeered: 11.11 %
# Ratio questions unansweered: 88.88 %
nrow(answer_info[answer_info$answered == 1, ]) / nrow(answer_info) * 100

# Answered Questions
answered <- answer_info[answer_info$answered == 1, ]
not_answered <- answer_info[answer_info$answered == 0, ]

# Tag Distribution accross questions
qTags <- as.character(sort(as.integer((unique(question_info$tag)))))
tag_distribution_accross_questions <- aggregate(question_info$tag, by=list(tag=question_info$tag), FUN = length)
names(tag_distribution_accross_questions) <- c("tags", "count")
plot(tag_distribution_accross_questions[ c("tags", "count") ], type="o")
summary(tag_distribution_accross_questions$count)

# Tag Distribution accross users
uTags <- as.character(sort(as.integer((unique(user_expert_info$expert_tag)))))
tag_distribution_accross_users <- aggregate(user_expert_info$expert_tag, by=list(tag=user_expert_info$expert_tag), FUN = length)
names(tag_distribution_accross_users) <- c("tags", "count")
plot(tag_distribution_accross_users[ c("tags", "count") ], type="o")
summary(tag_distribution_accross_users$count)

# Level of expertize
user_distribution_across_tags <- aggregate(user_expert_info$user_id, by=list(tag=user_expert_info$user_id), FUN = length)
names(user_distribution_across_tags) <- c("user_id", "tag_count")
expertize_distribution <- aggregate(user_distribution_across_tags$tag_count, by=list(tag=user_distribution_across_tags$tag_count), FUN = length)
names(expertize_distribution) <- c("number_of_tags", "number_of_users")
plot(expertize_distribution, type="o")

# Asked vs Answered
asked_to_user <- aggregate(answer_info$user_id, by=list(user_id=answer_info$user_id), FUN = length)
names(asked_to_user) <- c("user_id", "number_of_asked_questions")
answered_by_user <- aggregate(answered$user_id, by=list(user_id=answered$user_id), FUN = length)
names(answered_by_user) <- c("user_id", "number_of_answered_questions")

asked_vs_answered <- join(asked_to_user, answered_by_user, type="left", by=c("user_id"))
asked_vs_answered$number_of_answered_questions[ is.na(asked_vs_answered$number_of_answered_questions) ] <- 0
asked_vs_answered$ratio <- asked_vs_answered$number_of_answered_questions / asked_vs_answered$number_of_asked_questions

hist(asked_vs_answered$ratio,  xlim=c(0,1), breaks=25)
hist(asked_vs_answered[ asked_vs_answered$ratio > 0, ]$ratio, xlim=c(0,1), breaks=25)


# Question difficulty ( Ratio of top answers to answers )
question_info$top_answer_ratio <- (question_info$top_answers / question_info$answers)
easyQuestions <- function(tresh){
  easy_questions <- question_info[ question_info$top_answer_ratio >= tresh, ]$question_id
  easy_answers <- answer_info[answer_info$question_id %in% easy_questions, ]
  return(nrow(easy_answers[easy_answers$answered == 1, ]) / nrow(easy_answers))
}

# Questyion difficulty by type
answer_detailed_info <- join(answer_info, question_info, type="left", by=c("question_id"))

agg <- aggregate(answer_info, by=list(tag=answer_detailed_info$tag, user_id=answer_detailed_info$user_id, answered_count=answer_detailed_info$answered), FUN = length)