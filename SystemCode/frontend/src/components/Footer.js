import { ButtonGroup, Grid } from "@mui/material";
import Button from "@mui/material/Button";
import { useState } from "react";
import Tooltip from "@mui/material/Tooltip";
import SendIcon from "@mui/icons-material/Send";

const Footer = ({ onSend }) => {
  const [mBody, setBody] = useState();
  const onClickF = () => {
    console.log("button clicked");
    const message = mBody;
    onSend({ body: message });
    setBody("");
  };

  const handleChange = (e) => {
    console.log(e.target);
    setBody(e.target.value);
  };

  return (
    <Grid mt={3} container direction="row" alignItems="center">
      <Grid item xs={10}>
        <Button variant="contained" component="label">
          Upload
          <input
            hidden
            accept="image/*"
            multiple
            type="file"
            onChange={handleChange}
          />
        </Button>
      </Grid>
      <Grid item xs={2} alignItems="flex-end">
        <Grid container direction="column" alignContent="flex-end">
          <ButtonGroup orientation="vertical">
            <Tooltip title="Send my response">
              <Button
                variant="contained"
                onClick={onClickF}
                endIcon={<SendIcon />}
              >
                Send
              </Button>
            </Tooltip>
          </ButtonGroup>
        </Grid>
      </Grid>
    </Grid>
  );
};

export default Footer;
