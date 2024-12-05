import React from 'react';

const QuestionList = ({ questions }) => {
    return (
        <ul>
            {questions.map((question, index) => (
                <li key={index}>{question}</li>
            ))}
        </ul>
    );
};

export default QuestionList; 