import React from 'react';

const TopicList = ({ topics }) => {
    return (
        <ul>
            {topics.map((topic, index) => (
                <li key={index}>{topic}</li>
            ))}
        </ul>
    );
};

export default TopicList; 