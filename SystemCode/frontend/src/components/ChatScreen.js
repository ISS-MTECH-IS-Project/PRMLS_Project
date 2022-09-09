import { useState, useRef } from "react";
import Header from "./Header";
import Footer from "./Footer";
import Topbar from "./Topbar";
import Messages from "./Messages";
import { Box, Grid } from "@mui/material";
import Paper from "@mui/material/Paper";
import Divider from "@mui/material/Divider";
import moment from "moment";

// message : {body}
const ChatScreen = () => {
  const [messages, setMessages] = useState([]);

  // get next message
  const getNext = async (message) => {
    var mIndex = messages.length - 1;
    for (var i = messages.length - 1; i > 0; i--) {
      if (
        messages[i].symptoms !== undefined &&
        messages[i].symptoms.length > 0
      ) {
        mIndex = i;
        break;
      }
    }
    const m = messages[mIndex];
    const tempBody = m.body;
    m.body = message.body;
    const res = await fetch(`http://localhost:5000/api/classify`, {
      method: "POST",
      headers: {
        "Content-type": "application/json",
      },
      body: JSON.stringify(m),
    });
    const data = await res.json();
    m.body = tempBody;
    data.time = moment().format("hh:mm");
    setMessages([...messages, message, data]);
    scrollToBottom();
  };

  const sendMessage = (message) => {
    message.time = moment().format("hh:mm");
    setMessages([...messages, message]);
    getNext(message);
  };

  const messagesEndRef = useRef(null);
  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  // https://www.chatbot.com/chatbot-templates/
  return (
    <Grid container p={1} justifyContent="left">
      <Box display="none" sx={{ width: 0.18 }}>
        <Paper elevation={3}>
          <Grid item></Grid>
        </Paper>
      </Box>
      <Box display="inline-grid" sx={{ width: 0.6 }}>
        <Paper elevation={3}>
          <Grid item p={2}>
            <Header />
            <Topbar />
            <Messages messages={messages} />
            <Divider sx={{ mt: 2 }} />
            <Footer onSend={sendMessage} />
            <div ref={messagesEndRef} />
          </Grid>
        </Paper>
      </Box>
    </Grid>
  );
};

export default ChatScreen;
