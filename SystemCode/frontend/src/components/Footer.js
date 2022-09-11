import { ButtonGroup, Grid } from "@mui/material";
import Button from "@mui/material/Button";
import { useEffect, useState } from "react";
import Tooltip from "@mui/material/Tooltip";
import SendIcon from "@mui/icons-material/Send";

const Footer = ({ onSend }) => {
  const [image, setImage] = useState();
  const [preview, setPreview] = useState();
  const onClickF = () => {
    console.log("button clicked");
    onSend(image, preview);
    setImage(null);
  };

  const handleChange = (e) => {
    console.log(e.target);
    setImage(e.target.files[0]);
  };

  useEffect(() => {
    if (image) {
      const reader = new FileReader();
      reader.onloadend = () => {
        setPreview(reader.result);
      };
      reader.readAsDataURL(image);
    } else {
      setPreview(null);
    }
  }, [image]);

  return (
    <Grid mt={3} container direction="row" alignItems="center">
      <Grid item xs={10}>
        <Button variant="contained" component="label">
          Upload
          <input
            hidden
            accept="image/*"
            multiple={false}
            type="file"
            onChange={handleChange}
          />
        </Button>
        <img width={400} src={preview}></img>
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
