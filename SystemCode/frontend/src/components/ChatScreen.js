import { useState, useRef } from "react";
import Header from "./Header";
import Footer from "./Footer";
import Topbar from "./Topbar";
import Messages from "./Messages";
import { Box, Grid } from "@mui/material";
import Paper from "@mui/material/Paper";
import Divider from "@mui/material/Divider";
import moment from "moment";

const ChatScreen = () => {
  const [messages, setMessages] = useState([]);

  const uploadImg = async (file, preview) => {
    const reqBody = new FormData();
    reqBody.append("image", file, file.name);
    const res = await fetch(`http://localhost:5000/api/classify`, {
      method: "POST",
      body: reqBody,
    });
    const data = await res.json();
    setMessages([
      ...messages,
      { file: file, preview: preview, response: data },
    ]);
    scrollToBottom();
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
            <Footer onSend={uploadImg} />
            <div ref={messagesEndRef} />
          </Grid>
        </Paper>
      </Box>
    </Grid>
  );
};

export default ChatScreen;
